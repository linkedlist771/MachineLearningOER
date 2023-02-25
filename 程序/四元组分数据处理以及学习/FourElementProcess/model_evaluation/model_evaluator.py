import f
import pandas as pd

class ModelEvaluator:
    model = None
    perf = None
    dataloader = None
    predictor_leaderboard = None
    eval_info_df = None
    # predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred_test, auxiliary_metrics=True)

    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader
        y_pred_test = self.model.predictor.predict(dataloader.test)
        y_test = self.dataloader.y_test
        self.perf = model.eavluate_predictions(y_true=y_test, y_pred=y_pred_test, auxiliary_metrics=True)
        self.predictor_leaderboard = self.model.predictor.leaderboard(dataloader.test, silent=True)
        self.predictor_leaderboard = self.predictor_leaderboard.set_index(self.predictor_leaderboard["model"])
        self.predictor_leaderboard.loc[:, "score_val"] = -self.predictor_leaderboard.loc[:, "score_val"]
        self.predictor_leaderboard.loc[:, "score_test"] = -self.predictor_leaderboard.loc[:, "score_test"]

        #
        predictor = self.model.predictor
        all_models = predictor.get_model_names()
        model_to_use = all_models[-1]
        train_nolabel = dataloader.train_nolabel
        y_train = dataloader.y_train
        test_nolabel = dataloader.test_nolabel
        specific_model = self.model.predictor._trainer.load_model(model_to_use)
        eval_model_train = f.ModelEvaluation(specific_model, train_nolabel, y_train)
        eval_model_val = f.ModelEvaluation(specific_model, test_nolabel, y_test)
        all_models = self.predictor.get_model_names()
        model_to_use = all_models[-1]
        specific_model = self.predictor._trainer.load_model(model_to_use)
        import warnings
        warnings.filterwarnings('ignore')
        eval_info_df = pd.DataFrame(
            columns=["model", "R² train", "R² val", "MSE train", "MSE val", "RMSE train", "RMSE val", "MAE train",
                     "MAE val", "MAPE train", "MAPE val"])
        print(eval_info_df)
        for model in [*[predictor._trainer.load_model(i) for i in all_models[:-1]], predictor]:
            try:
                model_name = model.name
            except:
                model_name = "WeightedEnsemble_L2"
            eval_model_train = f.ModelEvaluation(model, train_nolabel, y_train)
            eval_model_val = f.ModelEvaluation(model, test_nolabel, y_test)
            # add row to end of DataFrame
            eval_info_df.loc[len(eval_info_df.index)] = [model_name,
                                                         eval_model_train.get_R2(), eval_model_val.get_R2(),
                                                         eval_model_train.get_MSE(), eval_model_val.get_MSE(),
                                                         eval_model_train.get_RMSE(), eval_model_val.get_RMSE(),
                                                         eval_model_train.get_MAE(), eval_model_val.get_MAE(),
                                                         eval_model_train.get_MAPE(), eval_model_val.get_MAPE(), ]
            print(f"model:{model_name}")
            print(f"R² train:{eval_model_train.get_R2()}")
            print(f"R² val:{eval_model_val.get_R2()}")
            print(f"MSE train:{eval_model_train.get_MSE()}")
            print(f"MSE val:{eval_model_val.get_MSE()}")
            print(f"RMSE train:{eval_model_train.get_RMSE()}")
            print(f"RMSE val:{eval_model_val.get_RMSE()}")
            print(f"MAE train:{eval_model_train.get_MAE()}")
            print(f"MAE val:{eval_model_val.get_MAE()}")
            print(f"MAPE train:{eval_model_train.get_MAPE()}")
            print(f"MAPE val:{eval_model_val.get_MAPE()}\n")
        self.eval_info_df = eval_info_df
        # 这里取， 几个表的结果。。。。。。。。。
        # this_train_x, this_train_y = X.iloc[train_index], y.iloc[train_index]  # 本组训练集
        # this_test_x, this_test_y = X.iloc[test_index], y.iloc[test_index]  # 本组验证集
        # for _ in zip(train_index, test_index):
        #     print(_)
        # # 训练本组的数据，并计算准确率
        # my_model.fit(this_train_x, this_train_y)
        # prediction = my_model.predict(this_test_x)
        # score = accuracy_score(this_test_y, prediction)
        # print(score)  # 得到预测结果区间[0,1]





