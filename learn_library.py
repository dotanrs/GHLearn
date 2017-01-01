import prepare
import display
import numpy as np
import tensorflow.contrib.learn.python.learn as learn
import tensorflow as tf
from sklearn import metrics
import random
import math

data, labels = prepare.get_data_labels_from_file('tester.csv')

im = np.reshape(data[0], (28, 28))

# Specify that all features have real-value data
# feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]
feature_columns = learn.infer_real_valued_columns_from_input(data)


# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.LinearClassifier(
                                            n_classes=10,
                                            feature_columns=feature_columns,
                                           )

classifier.fit(data,
               labels,
               steps=20)


test_data, test_labels = prepare.get_data_labels_from_file("test_no_header.csv")

predictions = list(classifier.predict(test_data,
                                           as_iterable=True))

score = metrics.accuracy_score(test_labels,
                               predictions)

print("Accuracy: %f" % score)


display.show9(test_data, predictions)

print(labels)
