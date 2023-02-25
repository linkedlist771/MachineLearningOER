from dataloader.DataLoader import DataLoader
from model.fusion_model import FusionModel
from model.custom_SVR import CustomSVR
from autogluon.tabular.models import TabTransformerModel


if __name__ == "__main__":
    DL = DataLoader()
    train_data, test_data = DL.train, DL.test
    custom_ResNet = None
    custom_TabTransformer = TabTransformerModel
    custom_model = {CustomSVR: {}, custom_TabTransformer: {'ag_args_fit': {'num_gpus': 0}}}
    use_GPU = True
    FM = FusionModel(label="10 mA cm‐2 ", save_path="fusion_model", custom_model=custom_model, use_GPU=use_GPU)
    FM.train(train_data)
    print(FM.predictor.leaderboard(test_data, silent=True))
