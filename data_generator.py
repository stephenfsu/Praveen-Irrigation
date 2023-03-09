# -*- coding: utf-8 -*-

import numpy as np
import keras

from dataset import Dataset


class DataGenerator(keras.utils.Sequence):

    """Custom generator for rfml 2d conv model

    Attributes:
        batch_size (int): batch_size
        indexes (list): index
        num_lines (int): num of lines in file
        raw_file (str): path/to/raw_file
        shuffle (bool): shuffle
    """

    def __init__(self, raw_file, batch_size=1000, img_shape=(512, 512),
                 shuffle=True):

        self.batch_size = batch_size
        self.raw_file = raw_file
        self.shuffle = shuffle
        self.num_lines = sum(1 for line in open(self.raw_file))
        self.list_IDs = [x for x in range(self.__len__())]
        self.shape = img_shape
        self.on_epoch_end()
        self.linsp = np.linspace(-3, 3, self.shape[0])
        self.ds = Dataset(self.raw_file)

    def __len__(self):
        """ length of generator"""
        return int(np.floor(self.num_lines / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_index = self.indexes[index]
        # Generate data
        X, y = self.__data_generation(batch_index)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.list_IDs
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def find_nearest_ids(self, array):
        idx = np.searchsorted(self.linsp, array, side="left") - 1
        return idx

    def __data_generation(self, batch_index):
        'Generates data containing batch_size samples'
        # Initialization
        ds = self.ds[self.batch_size * self.batch_index: self.batch_size * self.batch_index + 1]
        x_img = np.zeros((self.batch_size, self.shape[0], self.shape[1]),
                         dtype='uint8')
        ds = Dataset(self.raw_file, skip=self.batch_size * batch_index,
                     max_rows=self.batch_size)
        # Generate data
        for i, data in enumerate(ds.x_data):
            idx_real = self.find_nearest_ids(data[:, 0])
            idx_im = self.find_nearest_ids(data[:, 1])
            x_img[i,
                  idx_real,
                  idx_im
                  ] = 1

        return np.expand_dims(x_img, axis=-1), ds.y_data

