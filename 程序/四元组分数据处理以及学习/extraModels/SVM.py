import pandas as pd
from sklearn.utils import shuffle as reset
from custom_model import CustomModel
from sklearn.svm import SVR, SVC


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




if __name__ == '__main__':
    df = pd.read_excel(r"C:\Users\23174\Desktop\GitHub项目\毕设\MachineLearningOER\数据\四元数据添加Composition特征.xlsx")
    df = df.drop(["Ni", "Fe", "Co", "Ce","3mA cm‐2 ", "formula", "composition"], axis=1)
    features_to_drop = ["MagpieData minimum SpaceGroupNumber", "MagpieData maximum SpaceGroupNumber",
    "MagpieData range SpaceGroupNumber", "MagpieData mean SpaceGroupNumber", "MagpieData avg_dev SpaceGroupNumber",
                        "MagpieData mode SpaceGroupNumber"]
    df = df.drop(features_to_drop, axis=1)
    train, test = train_test_split(df, test_size=0.3,  random_state=1111)
    label = "10 mA cm‐2 "
    save = "四元组分10mAcm-2添加formula特征回归SVM"
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
    # import KNN lib from sklearn
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    for model in [SVR, KNeighborsRegressor, RandomForestRegressor, GradientBoostingRegressor]:
        custom_model = CustomModel(custom_model=model, problem_type="regression", params={})
        X_train = train.drop([label], axis=1)
        y_train = train[label]
        # We could also specify hyperparameters to override defaults
        # custom_model = CustomRandomForestModel(hyperparameters={'max_depth': 10})
        custom_model.fit(X=X_train, y=y_train)  # Fit custom model
        y_pred_test = custom_model.predict(test.drop([label], axis=1))  # Predict on test data
        y_pred_train = custom_model.predict(train.drop([label], axis=1))  # Predict on train data
        # calculate the mean absolute error
        from sklearn.metrics import mean_absolute_error
        import matplotlib.pyplot as plt
        mae_test = mean_absolute_error(test[label], y_pred_test)
        plt.figure(figsize=(12, 10))

        p0 = plt.plot(df[label], df[label], 'r--', linewidth=5)
        p1 = plt.scatter(y_pred_test, test[label], c='b', s=100)
        p2 = plt.scatter(train[label], y_pred_train, c='g', s=100)
        plt.legend(["Reference line", "Validating-set", "Training-set"], fontsize=15)
        plt.xlabel("ML Predict OP(mv)", fontsize=15)
        plt.ylabel("True OP/(mv)", fontsize=15)
        plt.title("Over Potential Prediction Graph", fontsize=15)
        plt.show()

        print(mae_test)
