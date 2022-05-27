import numpy as np
from sklearn.utils import shuffle
import pandas as pd


class Preprocess:
    """
    p is the number of predicted values in one entry
    t is the number of time steps in one entry
    valid_ratio is ratio of validation set
    """

    def __init__(self, p: int, t: int, split: bool, valid_ratio: float, scaling: str):
        self.train_dataset = 0
        self.mean = 0
        self.std = 0
        self.x_max = 0
        self.x_min = 0
        self.p = p
        self.t = t
        self.split = split
        self.valid_ratio = valid_ratio
        self.scaling = scaling

    def fit(self, dataset: pd.DataFrame):
        self.train_dataset = dataset
        if self.scaling == "min_max":
            self.x_max = np.max(dataset.values)
            self.x_min = np.min(dataset.values)
        elif self.scaling == "standard":
            self.mean = dataset.mean
            self.std = dataset.std
        return self

    def transform(self, dataset: pd.DataFrame):

        # Data scaling
        if self.scaling == "min_max":
            x_st = (dataset - self.x_min) / (self.x_max - self.x_min)
            dataset_scaled = x_st * (1 - (0)) + (0)
        elif self.scaling == "standard":
            dataset_scaled = dataset - self.mean / self.std
        elif self.scaling == "no":
            pass
        else:
            raise ValueError(
                "Defined scaling option {} is not available".format(self.scaling)
            )

        xs, indeces_x, ys, indeces_y = self.x_y_split(dataset_scaled)

        print("Number of time steps in one entry: ", self.t)
        print("Number of predictions in one entry: ", self.p)
        print("Total length of one entry: ", self.t + self.p)
        print("Total number of samples: ", len(xs))

        if self.split:
            (
                x_train,
                x_valid,
                y_train,
                y_valid,
                x_train_indeces,
                x_valid_indeces,
                y_train_indeces,
                y_valid_indeces,
            ) = self.train_valid_split(xs, indeces_x, ys, indeces_y)

            print("Training dataset shape: ", x_train.shape)
            print("Validation dataset shape: ", x_valid.shape)
            print("Training labels shape: ", y_train.shape)
            print("Validation labels shape: ", y_valid.shape)

            return (
                x_train,
                y_train,
                x_valid,
                y_valid,
                x_train_indeces,
                x_valid_indeces,
                y_train_indeces,
                y_valid_indeces,
            )

        else:

            print("Training dataset shape: ", xs.shape)
            print("Training labels shape: ", ys.shape)

            return xs, ys, indeces_x, indeces_y

    def x_y_split(self, dataset: pd.DataFrame):

        indeces = dataset.index
        dataset = dataset.values.flatten()

        w = dataset.shape[0]
        entry = self.t + self.p
        m = w - entry + 1

        grouped = dataset[0:entry]
        grouped_indeces = indeces[0:entry]
        for i in range(1, m):
            grouped = np.vstack([grouped, dataset[i:entry + i]])
            grouped_indeces = np.vstack([grouped_indeces, indeces[i:entry + i]])

        x = grouped[:, 0:self.t]
        x = x.reshape((x.shape[0], x.shape[1], 1))
        indeces_x = grouped_indeces[:, 0:self.t]
        y = grouped[:, -self.p:]
        indeces_y = grouped_indeces[:, -self.p:]

        return x, indeces_x, y, indeces_y

    def train_valid_split(
        self, x: np.ndarray, indeces_x: np.ndarray, y: np.ndarray, indeces_y: np.ndarray
    ):
        valid_number = int(len(x) * self.valid_ratio)
        x, y = shuffle(x, y, random_state=42)
        x_valid, y_valid = x[-valid_number:, :, :], y[-valid_number:, :]
        x_train, y_train = x[:-valid_number, :, :], y[:-valid_number, :]
        x_valid_indeces, y_valid_indeces = (
            indeces_x[-valid_number:],
            indeces_y[-valid_number:],
        )
        x_train_indeces, y_train_indeces = (
            indeces_x[:-valid_number],
            indeces_y[:-valid_number],
        )

        return (
            x_train,
            x_valid,
            y_train,
            y_valid,
            x_train_indeces,
            x_valid_indeces,
            y_train_indeces,
            y_valid_indeces,
        )
