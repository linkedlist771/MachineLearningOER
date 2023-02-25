import pandas
import numpy as np
from sklearn.utils import shuffle as reset

class DataLoader:

    def __init__(self, path):
        self.path = path
        if path.endswith('.csv'):
            self.data = pandas.read_csv(path)
        elif path.endswith('.xlsx'):
            self.data = pandas.read_excel(path)
        else:
            raise Exception('File format not supported.')


    def train_test_split(self, test_size=0.3, shuffle=True, random_state=None):
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
        data = self.data.copy()
        if shuffle:
            data = reset(data, random_state=random_state)

        train = data[int(len(data) * test_size):].reset_index(drop=True)
        test = data[:int(len(data) * test_size)].reset_index(drop=True)

        return train, test



