import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn.python.learn as learn
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, preprocessing
import time
import numpy as np
from PIL import Image
import tensorflow.contrib.learn.python.learn as learn
from numpy import genfromtxt
from settings import *
import math
import random

def get_data_labels_from_file(filename):
    """ assuming the labels column is the last one """

    file = genfromtxt(filename, delimiter=',', dtype=float)

    # remove headers
    headers = file[0]
    file = file[1:]

    # put last column in labels, rest in data
    data = np.array([row[:-1] for row in file], dtype=float)
    labels = np.array([int(row[-1]) for row in file], dtype=int)

    return data, labels


class DataSets:
    
    def __init__(self, train_file = TRAIN_FILE, test_file = TEST_FILE):
        self.train = DataSet(train_file)
        self.test = DataSet(test_file)

        if DO_REDUCE_MEAN:
            means = self.train.reduce_mean()
            self.test.reduce_mean(means)

class DataSet:
    
    def __init__(self, file):
        self.data, self.labels = get_data_labels_from_file(file)
        self.labels = self.labels
        self.batch_offset = 0
        self.num_examples = len(self.data)
        self.batch_size = BATCH_SIZE
        
    def next_batch(self):
        select = [random.randint(0, self.num_examples - 1) for _ in range(self.batch_size)]
        batch_data   = [self.data[idx]   for idx in select]
        batch_labels = [self.labels[idx] for idx in select]

        self.batch_offset += self.batch_size
        return batch_data, batch_labels
    
    def custom_batch(self, num):
        batch_data   = [self.data[num]   for _ in range(self.batch_size)]
        batch_labels = [self.labels[num] for _ in range(self.batch_size)]

        self.batch_offset += self.batch_size
        return batch_data, batch_labels

    def next_batch_iterative(self):
        if self.batch_offset + self.batch_size >= self.num_examples:
            self.batch_size = self.num_examples - self.batch_offset
            
        if self.batch_offset > self.num_examples or self.batch_size == 0:
            return 0, 0
        batch_data   = self.data   [self.batch_offset: self.batch_offset + self.batch_size]
        batch_labels = self.labels [self.batch_offset: self.batch_offset + self.batch_size]

        self.batch_offset += self.batch_size
        return batch_data, batch_labels
    
    def array_to_one_hot(self, arr):
        l = len(arr)
        one_hot_labels = np.zeros((l, 10))
        one_hot_labels[np.arange(l), arr] = 1
        return one_hot_labels
    
    def reduce_mean(self, means = None):
        if means is None:
            means = np.mean(self.data, axis = 0)
        
        self.data -= means
        return means
