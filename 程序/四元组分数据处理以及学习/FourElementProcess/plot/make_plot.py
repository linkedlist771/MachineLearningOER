import numpy as np
import os

class ModelPlotter:

    def __init__(self, model, dataloader, model_evaluator, plot_dir_path):
        import matplotlib.pyplot as plt
        # change into the plot directory and make plot in it
        if not os.path.exists(plot_dir_path):
            os.mkdir(plot_dir_path)
        os.chdir(plot_dir_path)
        predictor_leaderboard = model_evaluator.predictor_leaderboard
        predictor_leaderboard.plot.bar(figsize=(20, 16))
        # set the size of the xlable and ylabel and the legend
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.xlabel('predictor', fontsize=30)
        plt.xticks(fontsize=15)
        plt.ylabel('score', fontsize=30)
        plt.legend(fontsize=20)
        plt.savefig("特征抽取后的各个模型的结果图", dpi=300)
        plt.show()
        y_pred = dataloader.y_pred
        train = dataloader.train
        label = dataloader.label
        y_test = dataloader.y_test
        y_pred_train = model.predict(train)
        max_op = np.max(y_pred.values)
        min_op = np.min(y_pred.values)
        x = np.arange(min_op, max_op, 0.01)
        y = x
        plt.figure(figsize=(12, 10))
        y_pred_test = model.predictor.predict(dataloader.test)
        p0 = plt.plot(x, y, 'r--', linewidth=5)
        p1 = plt.scatter(y_pred_test.values, y_test.values)
        p2 = plt.scatter(train[label].values, y_pred_train)
        plt.legend(["Reference line", "Validating-set", "Training-set"], fontsize=15)
        plt.xlabel("ML Predict OP(mv)", fontsize=15)
        plt.ylabel("True OP/(mv)", fontsize=15)
        plt.title("Over Potential Prediction Graph", fontsize=15)
        plt.savefig("抽取特征的结果图.png", dpi=300)

        test_differ = y_pred_test.values - y_test.values
        train_differ = train[label].values - y_pred_train

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

        # An "interface" to matplotlib.axes.Axes.hist() method
        n, bins, patches = plt.hist(x=test_differ, bins='auto',
                                    alpha=0.8, rwidth=0.85)
        n, bins, patches = plt.hist(x=train_differ, bins='auto',
                                    alpha=0.2, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('$OP_{ML}-OP_{true}$', fontsize=15)
        plt.ylabel('Counts', fontsize=15)
        plt.legend(["Validating-set", "Training-set"], fontsize=15)
        plt.title("Over Potential Prediction Distribution", fontsize=15)
        plt.subplots_adjust(bottom=0.2, top=0.9)
        plt.savefig("抽取特征的结果分布图.png", dpi=300)
        # Set a clean upper y-axis limit.
        # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        eval_info_df_list_mean = model_evaluator.eval_info_df

        eval_info_df_list_mean.set_index(eval_info_df_list_mean['model'], inplace=True)
        eval_info_df_list_mean.plot.bar(figsize=(20, 12))
        plt.subplots_adjust(bottom=0.2, top=0.9)

        plt.xlabel('predictor', fontsize=30)
        plt.xticks(fontsize=15)
        plt.ylabel('score', fontsize=30)
        plt.legend(fontsize=20)
        plt.savefig("5-折交叉验证 特征抽取后的各个模型的误差图", dpi=300)
        predicitions = eval_info_df_list_mean.drop(
            columns=["R² train", "R² val", "MSE train", "MSE val", "MAE train", "MAE val", "MAPE train", "MAPE val"])
        predicitions.columns = ['model', 'Training-set', 'Validating-set']
        predicitions.set_index(predicitions['model'], inplace=True)
        predicitions.plot.bar(figsize=(20, 12))
        plt.subplots_adjust(bottom=0.2, top=0.9)

        plt.xlabel('predictor', fontsize=30)
        plt.xticks(fontsize=15)
        plt.ylabel("RMSE", fontsize=30)
        plt.legend(fontsize=20)
        plt.savefig("5-折交叉验证 抽取特征的RMSE.png", dpi=300)

        # MAPE
        predicitions = eval_info_df_list_mean.drop(
            columns=["R² train", "R² val", "MSE train", "MSE val", "MAE train", "MAE val", "RMSE train", "RMSE val"])
        predicitions.columns = ['model', 'Training-set', 'Validating-set']
        predicitions.set_index(predicitions['model'], inplace=True)
        predicitions.plot.bar(figsize=(20, 12))
        plt.subplots_adjust(bottom=0.2, top=0.9)

        plt.xlabel('predictor', fontsize=30)
        plt.xticks(fontsize=15)
        plt.ylabel("MAPE/%", fontsize=30)
        plt.legend(fontsize=20)
        plt.savefig("5-折交叉验证 抽取特征的MAPE.png", dpi=300)
