# -*- coding: utf-8 -*-
"""
This code is written by Ibnu Hafidz Arief for anyone in the universe
"""


import numpy as np
import pandas as pd


class kriging():
    def __init__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        # check for data format
        x, y = self.checkfordatatypes__(x, y)
        # check inputs dimension
        x, y = self.checkinputsdimension__(x, y)
        # check if there is a duplication in the input x
        x, y = self.checkforduplication__(x, y)
        # scale the inputs from 0-1
        self.xs, self.ys, self.min_x, self.max_x,
        self.min_y, self.max_y. self.y_log = self.scale_trainingdata(x, y)

    def checkinputsdimension__(self, x, y):
        '''
        This function checks for the dimension of the input data
        if a condition is not met then return an error
        '''
        if len(x.shape) == 1:
            # then it only contains 1 feature
            # reshape the array into a 2D
            x = x.reshape((x.shape[0], 1))
        if len(y.shape) == 2:
            # it should be one dimensional array
            y = y.reshape(y.shape[0])
        assert (x.shape[0] != y.shape[0]),
        "The number of data in x must be the same with y"
        return x, y

    def checkfordatatypes__(self, x, y):
        try:
            x = x.astype(np.float)
            y = y.astype(np.float)
            return x, y
        except:
            raise ValueError("The data must be float. String is not allowed")

    def checkforduplication__(self, x, y):
        '''
        This function checks for a duplication of x
        if found then delete those lines and take the first record
        '''
        unique_x, unique_idx = np.unique(x, axis=0)
        unique_y = np.take(y, unique_idx)
        return unique_x, unique_y

    def scale_trainingdata(self, x, y):
        '''
        This function scales both x and y into 0-1
        '''
        xs = np.copy(x)
        min_x = np.zeros(x.shape[1])
        max_x = np.zeros(x.shape[1])

        if max(y) / min(y) > 50:
            y_log = True
            y = np.log10(y)
        else:
            y_log = False

        min_y = min(y)
        max_y = max(y)

        ys = (y - min_y) / (max_y - min_y)
        for i in range(x.shape[1]):
            min_x[i] = min(x[:, i])
            max_x[i] = max(x[:, i])

            xs[:, i] = (x[:, i] - min_x[i]) / (max_x[i] - min_x[i])

        return xs, ys, min_x, max_x, min_y, max_y, y_log

    def train_model(self, xs, ys):
        '''
        This function creates a kriging model by training the data
        '''
        return

    def predict(self, xtest):
        '''
        This functions predict the test data from the trained model
        '''
        # check the xtest dimension
        xtest, ydummy = self.checkfordatatypes__(xtest, xtest[:, 0])
        xtest, ydummy = self.checkinputsdimension__(xtest, xtest[:, 0])
        ydummy = 0  # release memory
        # scale the xtest
