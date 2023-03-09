# -*- coding: utf-8 -*-
# Need to change this file and add code to the classes for loading the data into
# the network

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical

import numpy as np


class Dataset:

    def __init__(self, data_path, num_classes, data_type = '2d', skip=0, max_rows=None):
        """
        """
        self.num_classes = num_classes
        self.data_path = data_path
        self.skip = skip
        self.max_rows = max_rows
        self.read_data()
        self.process_data()

    def read_data(self):
        """read data from file
        """
        # rf_dtype = [("raw-iq" + str(i), "c16") for i in range(1024)]
        # rf_dtype += [("mod", "U10"), ("freq", "i8"),
        #              ("rise-time", "f16"), ("trans-id", "U10")]

        # if self.max_rows:
        #     self.data_holder = np.loadtxt(
        #         self.data_path, dtype=rf_dtype, skiprows=self.skip,
        #         max_rows=self.max_rows)
        # else:
        #     print(self.data_path)
        #     self.data_holder = np.loadtxt(
        #         self.data_path, dtype=rf_dtype, skiprows=self.skip)

        self.data_holder = np.load(self.data_path)

    def process_data(self, data_type = '2d'):
        """
        Args:
            data_type (str, optional): Description

        """

    def normalize_data(self, data):
        """
        Desc : Normalize the data with zero mean and unit variance for
               each column, for I values
        """

        standard = preprocessing.StandardScaler().fit(data)
        data = standard.transform(data)
        return data

    def split_data(self, validation_split=True, split_ratio=0.2):
        """
        """
        if validation_split:
            raise NotImplementedError
        else:
            return self.x_data, self.y_data


class HelicalDataset(Dataset):

    def __init__(self, data_path, num_classes, time_dim, time_step, raster_dim):
        super().__init__(data_path=data_path, num_classes=num_classes)
        self.generate_data(time_step, time_dim, raster_dim)

    def find_nearest_ids(self, array, linsp):
        idx = np.searchsorted(linsp, array, side="left") - 1
        return idx

    def generate_data(self, time_step=10, time_dim=150, raster_dim=128, ):
        """make dataset for 3d convolutions
        """



class ColorCodedDataset(Dataset):
	pass
