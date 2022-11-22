###### 这个文件是用于补充在毕设过程中经常用到的函数捏
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
class ModelEvaluation():
    def __init__(
        self,
        model: object,
        X,
        y_true) -> None:
        self.model = model
        self.X = X
        self.y_true = y_true
        self.y_pred = model.predict(X)
    def get_y_pred(self):
        return self.y_pred
    def get_MAE(self)->float:
        return mean_absolute_error(self.y_true, self.y_pred)
    def get_MSE(self)->float:
        return mean_squared_error(self.y_true, self.y_pred)
    def get_RMSE(self)->float:
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))
    def get_R2(self)->float:
        return r2_score(self.y_true, self.y_pred)
    def get_MAPE(self)->float:
        return np.mean(np.abs((self.y_true-self.y_pred) / self.y_true)) * 100