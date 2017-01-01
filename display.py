import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn.python.learn as learn
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, preprocessing
import time
import numpy as np
from PIL import Image
import tensorflow.contrib.learn.python.learn as learn
import matplotlib.pyplot as plt
import matplotlib.colors
import random

def displayImages(images, titles, shape):
    size = len(images)
    for i in range(size):
        plt.subplot(int(str(shape) + str(i + 1)))
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def displayImage(image, title = ''):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()

def print_prediction(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.

    return logits[0]

def show9(data, labels, select = None):
    # print(select)
    
    if select == None:
        select = [random.randint(0, len(data) - 1) for _ in range(9)]
    
    res_images = []
    res_titles = []
    num = len(select)

    for i in range(9):
        if i >= num:
            break
        a = np.reshape(data[select[i]], (28, 28))
        res_images.append(a)
        res_titles.append(str(select[i]) + ": " + str(labels[select[i]]))
    #     displayImage(a, iris_predictions[i])
    displayImages(res_images, res_titles, 33)