from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


class Preprocess:
    """
    p is the number of predicted values in one entry
    t is the number of time steps in one entry
    valid_ratio is ratio of validation set
    """

    def __init__(self, p=5, t=25, valid_ratio=0.15):
        self.train_dataset = None
        self.mean = None
        self.std = None
        self.p = p
        self.t = t
        self.valid_ratio = valid_ratio

    def fit(self, dataset):
        self.train_dataset = dataset
        self.mean = np.mean(dataset, axis=0)
        self.std = np.std(dataset, axis=0)
        return self

    def transform(self, dataset):

        # Data scaling
        dataset_scaled = dataset - self.mean / self.std

        x = {}
        y = {}
        for price in dataset.columns.to_list():
            xs, indeces_x, ys, indeces_y = self.x_y_split(dataset[price])
            (
                x[price + "_train"],
                x[price + "_valid"],
                y[price + "_train"],
                y[price + "_valid"],
                x_train_indeces, 
                x_valid_indeces,
                y_train_indeces, 
                y_valid_indeces
            ) = self.train_valid_split(xs, indeces_x, ys, indeces_y)

        print("Number of time steps in one entry: ", self.t)
        print("Number of predictions in one entry: ", self.p)
        print("Total length of one entry: ", self.t + self.p)
        print("Total number of samples: ", len(xs))
        print("Number of samples in training set: ", len(x["Close_train"]))
        print("Number of samples in validation set: ", len(x["Close_valid"]))

        x_train = np.empty((x["Close_train"].shape[0], x["Close_train"].shape[1], 4))
        x_valid = np.empty((x["Close_valid"].shape[0], x["Close_valid"].shape[1], 4))
        y_train = np.empty((y["Close_train"].shape[0], y["Close_train"].shape[1], 4))
        y_valid = np.empty((y["Close_valid"].shape[0], y["Close_valid"].shape[1], 4))

        for i in range(len(dataset.columns.to_list())):
            x_train[:, :, i] = x[price + "_train"]
            x_valid[:, :, i] = x[price + "_valid"]
            y_train[:, :, i] = y[price + "_train"]
            y_valid[:, :, i] = y[price + "_valid"]

        print("Training dataset shape: ", x_train.shape)
        print("Validation dataset shape: ", x_valid.shape)
        print("Training labels shape: ", y_train.shape)
        print("Validation labels shape: ", y_valid.shape)

        return x_train, y_train, x_valid, y_valid, x_train_indeces, x_valid_indeces, y_train_indeces, y_valid_indeces

    def x_y_split(self, dataset):

        indeces = dataset.index

        w = dataset.shape[0]
        entry = self.t + self.p
        m = w - entry + 1

        grouped = dataset[0:entry]
        grouped_indeces = indeces[0:entry]
        for i in range(1, m):
            grouped = np.vstack([grouped, dataset[i : entry + i]])
            grouped_indeces = np.vstack([grouped_indeces, indeces[i : entry + i]])

        x = grouped[:, 0 : self.t]
        indeces_x = grouped_indeces[:, 0 : self.t]
        y = grouped[:, (-self.p) :]
        indeces_y = grouped_indeces[:, (-self.p) :]

        return x, indeces_x, y, indeces_y

    # Convert x shaped (m, t, 4) and y shaped (n, p, 4) into dataset (m + n + t + p, 4)

    def x_y_split_inverse(self, x, y, indeces_x, indeces_y):
        prices = ["Open", "High", "Low", "Close"]
        dataset = pd.dataFrame()
        return dataset

    def train_valid_split(self, x, indeces_x, y, indeces_y):
        valid_number = int(len(x) * self.valid_ratio)
        x_valid, y_valid = x[-valid_number:], y[-valid_number:]
        x_train, y_train = x[:-valid_number], y[:-valid_number]
        x_valid_indeces, y_valid_indeces = indeces_x[-valid_number:], indeces_y[-valid_number:]
        x_train_indeces, y_train_indeces = indeces_x[:-valid_number], indeces_y[:-valid_number]

        return x_train, x_valid, y_train, y_valid, x_train_indeces, x_valid_indeces, y_train_indeces, y_valid_indeces

    def stack_data(self, dataset):
        stacked = np.dstack
        return stacked
