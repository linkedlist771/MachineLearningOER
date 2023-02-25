from autogluon.tabular import TabularPredictor
import os

class FusionModel:

    label = None
    problem_type = "regression"
    save_path = None
    model = None
    predictor = None
    custom_model = None

    def __init__(self, label, save_path, custom_model=None, use_GPU=False, load_model=False):
        """
        :param label: 标签
        :param save_path: 模型保存路径
        :param custom_model: 自定义模型
        :param use_GPU: 是否使用GPU
        :param load_model: 是否加载模型
        """
        self.label = label
        self.save_path = save_path
        self.model = TabularPredictor(label=label, problem_type=self.problem_type, path=save_path)
        self.custom_model = custom_model
        self.use_GPU = use_GPU
        if load_model:
            if os.path.exists(save_path):
                self.predictor = self.model.load(save_path)
            else:
                raise FileNotFoundError("模型权重路径不存在， 请检查路径是否正确！")

    def load(self):
        self.predictor = self.model.load(self.save_path)

    def train(self, train_data):
        if self.custom_model is not None:
            from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
            custom_hyperparameters = get_hyperparameter_config('default')
            for custom_model, hyperparameters in self.custom_model.items():
                custom_hyperparameters[custom_model] = hyperparameters
            # Train the default models plus a single tuned CustomRandomForestModel
            if self.use_GPU:
                self.predictor = self.model.fit(train_data, ag_args_fit={'ag_args_fit': {'num_gpus': 0}}, hyperparameters=custom_hyperparameters, )
            else:
                self.predictor = self.model.fit(train_data, hyperparameters=custom_hyperparameters)

        else:
            if self.use_GPU:
                self.predictor = self.model.fit(train_data,  ag_args_fit={'ag_args_fit': {'num_gpus': 0}},)
            else:
                self.predictor = self.model.fit(train_data,)

    def predict(self, data):
        self.predictor.predict(data)


if __name__ == "__main__":
    # how to import the dataloadr from the father directory
    pass