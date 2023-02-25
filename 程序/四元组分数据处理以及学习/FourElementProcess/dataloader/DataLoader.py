import pandas as pd
from sklearn.utils import shuffle as reset


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


class DataLoader:

    """"
    这里可以导入不同的数据集，然后进行数据处理，包括数据的划分，特征的选择等等
    """
    # 全部使用public属性，不使用private属性
    data_path = r"C:\Users\23174\Desktop\GitHub项目\毕设\MachineLearningOER\数据\四元数据添加Composition特征.xlsx"
    columns_to_drop = ["Ni", "Fe", "Co", "Ce", "3mA cm‐2 ", "formula", "composition", "MagpieData minimum SpaceGroupNumber", "MagpieData maximum SpaceGroupNumber",
                       "MagpieData range SpaceGroupNumber", "MagpieData mean SpaceGroupNumber", "MagpieData avg_dev SpaceGroupNumber",
                       "MagpieData mode SpaceGroupNumber"]
    label = "10 mA cm‐2 "
    df = None
    train = None
    test = None
    test_nolabel = None
    train_no_label = None
    y_train = None
    y_test = None

    def __init__(self):
        """"
        所有的处理在初始化时来完成。"""
        self.df = pd.read_excel(self.data_path)
        self.df = self.df.drop(self.columns_to_drop, axis=1)
        self.train, self.test = train_test_split(self.df, test_size=0.3, random_state=1111)
        self.test_nolabel = self.test.drop(columns=[self.label])
        self.train_nolabel = self.train.drop(columns=[self.label])
        self.y_train = self.train[self.label]
        self.y_test = self.test[self.label]

if __name__ == "__main__":
    DL = DataLoader()
    for attr in dir(DL):
        print(attr, getattr(DL, attr))