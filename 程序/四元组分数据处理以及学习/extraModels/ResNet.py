import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.utils import shuffle as reset
from custom_model import CustomModel
from sklearn.svm import SVR, SVC
import rtdl

def train_test_split(data, test_size=0.3, shuffle=True, random_state=None):
    '''Split DataFrame into random train and test subsets

    Parameters
    ----------
    data : pandas dataframe, need to split dataset.

    test_size : float
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    shuffle : boolean, optional (default=None)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    '''
    if shuffle:
        data = reset(data, random_state=random_state)

    train = data[int(len(data) * test_size):].reset_index(drop=True)
    test = data[:int(len(data) * test_size)].reset_index(drop=True)

    return train, test


ModuleType = Union[str, Callable[..., nn.Module]]
def _make_nn_module(module_type: ModuleType, *args) -> nn.Module:
    if isinstance(module_type, str):
        try:
            cls = getattr(nn, module_type)
        except AttributeError as err:
            raise ValueError(
                f'Failed to construct the module {module_type} with the arguments {args}'
            ) from err
        return cls(*args)
    else:
        return module_type(*args)


class ResNet(nn.Module):
    """The ResNet model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

        ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)

                 |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
                 |                                                                  |
         Block: (in) ------------------------------------------------------------> Add -> (out)

          Head: (in) -> Norm -> Activation -> Linear -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `ResNet`."""

        def __init__(
            self,
            *,
            d_main: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout_first: float,
            dropout_second: float,
            normalization: ModuleType,
            activation: ModuleType,
            skip_connection: bool,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x: Tensor) -> Tensor:
            x_input = x
            x = self.normalization(x)
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            if self.skip_connection:
                x = x_input + x
            return x

    class Head(nn.Module):
        """The final module of `ResNet`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            normalization: ModuleType,
            activation: ModuleType,
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)

        def forward(self, x: Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)
            return x

    def __init__(
        self,
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        normalization: ModuleType,
        activation: ModuleType,
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()

        self.first_layer = nn.Linear(d_in, d_main)
        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main=d_main,
                    d_hidden=d_hidden,
                    bias_first=True,
                    bias_second=True,
                    dropout_first=dropout_first,
                    dropout_second=dropout_second,
                    normalization=normalization,
                    activation=activation,
                    skip_connection=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = ResNet.Head(
            d_in=d_main,
            d_out=d_out,
            bias=True,
            normalization=normalization,
            activation=activation,
        )

    @classmethod
    def make_baseline(
        cls: Type['ResNet'],
        *,
        d_in: int,
        n_blocks: int,
        d_main: int,
        d_hidden: int,
        dropout_first: float,
        dropout_second: float,
        d_out: int,
    ) -> 'ResNet':
        """Create a "baseline" `ResNet`.

        This variation of ResNet was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * :code:`Norm` = :code:`BatchNorm1d`

        Args:
            d_in: the input size
            n_blocks: the number of Blocks
            d_main: the input size (or, equivalently, the output size) of each Block
            d_hidden: the output size of the first linear layer in each Block
            dropout_first: the dropout rate of the first dropout layer in each Block.
            dropout_second: the dropout rate of the second dropout layer in each Block.

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        return cls(
            d_in=d_in,
            n_blocks=n_blocks,
            d_main=d_main,
            d_hidden=d_hidden,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            normalization='BatchNorm1d',
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class MLP(nn.Module):
    """The MLP model used in [gorishniy2021revisiting].

    The following scheme describes the architecture:

    .. code-block:: text

          MLP: (in) -> Block -> ... -> Block -> Linear -> (out)
        Block: (in) -> Linear -> Activation -> Dropout -> (out)

    Examples:
        .. testcode::

            x = torch.randn(4, 2)
            module = MLP.make_baseline(x.shape[1], [3, 5], 0.1, 1)
            assert module(x).shape == (len(x), 1)

    References:
        * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
    """

    class Block(nn.Module):
        """The main building block of `MLP`."""

        def __init__(
            self,
            *,
            d_in: int,
            d_out: int,
            bias: bool,
            activation: ModuleType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
        self,
        *,
        d_in: int,
        d_layers: List[int],
        dropouts: Union[float, List[float]],
        activation: Union[str, Callable[[], nn.Module]],
        d_out: int,
    ) -> None:
        """
        Note:
            `make_baseline` is the recommended constructor.
        """
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)

        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=d_layers[i - 1] if i else d_in,
                    d_out=d,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
            ]
        )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
        cls: Type['MLP'],
        d_in: int,
        d_layers: List[int],
        dropout: float,
        d_out: int,
    ) -> 'MLP':
        """Create a "baseline" `MLP`.

        This variation of MLP was used in [gorishniy2021revisiting]. Features:

        * :code:`Activation` = :code:`ReLU`
        * all linear layers except for the first one and the last one are of the same dimension
        * the dropout rate is the same for all dropout layers

        Args:
            d_in: the input size
            d_layers: the dimensions of the linear layers. If there are more than two
                layers, then all of them except for the first and the last ones must
                have the same dimension. Valid examples: :code:`[]`, :code:`[8]`,
                :code:`[8, 16]`, :code:`[2, 2, 2, 2]`, :code:`[1, 2, 2, 4]`. Invalid
                example: :code:`[1, 2, 3, 4]`.
            dropout: the dropout rate for all hidden layers
            d_out: the output size
        Returns:
            MLP

        References:
            * [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko, "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        assert isinstance(dropout, float), 'In this constructor, dropout must be float'
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'In this constructor, if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            activation='ReLU',
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.head(x)
        return x





if __name__ == "__main__":
    # UserWarning: torch.range is deprecated and will be removed in a future release because its behavior is inconsistent with Python's range builtin. Instead, use torch.arange, which produces values in [start, end).
    #   x = torch.range(0, 10).reshape(-1, 1)
    # x = torch.arange(0, 20, 0.01).reshape(-1, 1).float()
    # y = torch.sin(x)
    df = pd.read_excel(r"C:\Users\23174\Desktop\GitHub项目\毕设\MachineLearningOER\数据\四元数据添加Composition特征.xlsx")
    df = df.drop(["Ni", "Fe", "Co", "Ce","3mA cm‐2 ", "formula", "composition"], axis=1)
    features_to_drop = ["MagpieData minimum SpaceGroupNumber", "MagpieData maximum SpaceGroupNumber",
    "MagpieData range SpaceGroupNumber", "MagpieData mean SpaceGroupNumber", "MagpieData avg_dev SpaceGroupNumber",
                        "MagpieData mode SpaceGroupNumber"]
    df = df.drop(features_to_drop, axis=1)
    train, test = train_test_split(df, test_size=0.3,  random_state=1111)
    label = "10 mA cm‐2 "
    X_train = train.drop([label], axis=1)
    y_train = train[label]
    X_train = torch.Tensor(X_train.values)
    y_train = torch.Tensor(y_train.values)
    X_test = torch.Tensor(test.drop([label], axis=1).values)
    y_test = torch.Tensor(test[label].values)

    # module = ResNet.make_baseline(
    #     d_in=X_train.shape[1],
    #     d_main=256,
    #     d_hidden=512,
    #     dropout_first=0.2,
    #     dropout_second=0.0,
    #     n_blocks=2,
    #     d_out=1,
    # )
    module = MLP.make_baseline(
        d_in=X_train.shape[1],
        d_layers=[256, 256, 256],
        dropout=0.2,
        d_out=1,
    )
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    print(module(X_train))
    batch_size = 620
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = module.to(device)
    x = X_train.to(device)
    y = y_train.to(device)
    x_test = X_test.to(device)
    y_test = y_test.to(device)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-4)
    for epoch in range(10000):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = module(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        loss_trian = torch.nn.functional.mse_loss(module(x), y)
        loss_test = torch.nn.functional.mse_loss(module(x_test), y_test)
        print(f"Epoch {epoch}: train loss{loss.item()}")
        print(f"Epoch {epoch}: test loss{loss_trian.item()}")
    # : can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
    # plt.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy(), label='ground truth')
    # plt.plot(x.cpu().detach().numpy(), module(x).cpu().detach().numpy(), label='prediction')


    y_pred = module(torch.Tensor(df.drop([label], axis=1).values).to(device)).cpu().detach().numpy()
    import numpy as np
    y_pred_test = module(x_test).cpu().detach().numpy()
    y_pred_train = module(x).cpu().detach().numpy()
    max_op = np.max(y_pred)
    min_op = np.min(y_pred)
    x = np.arange(min_op, max_op, 0.01)
    y = x
    plt.figure(figsize=(12, 10))

    p0 = plt.plot(x, y, 'r--', linewidth=5)
    p1 = plt.scatter(y_pred_test, y_test.cpu().detach().numpy())
    p2 = plt.scatter(train[label], y_pred_train)
    plt.legend(["Reference line", "Validating-set", "Training-set"], fontsize=15)
    plt.xlabel("ML Predict OP(mv)", fontsize=15)
    plt.ylabel("True OP/(mv)", fontsize=15)
    plt.title("Over Potential Prediction Graph", fontsize=15)
    plt.show()

    print("ok")