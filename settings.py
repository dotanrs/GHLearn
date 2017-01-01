import math
import tensorflow as tf
import utils

# Input
TRAIN_FILE = "train_medium.csv"
TEST_FILE = "eval_medium.csv"
DO_REDUCE_MEAN = True;
NUM_CLASSES = 10
IMAGE_SIZE = 28

# Output
PRINT_SUMMARIES = True
CHECKPOINT_INTERVAL = 500
SUMMARY_INTERVAL = 10
FILENAME_TIMESTAMP = True
OUTPUT_PATH = "outputs/output.csv"
COLUMN_NAMES = ["ImageId", "Label"]
LOG_DIR = '/tmp/tensorflow/mnist/logs/fully_connected_feed'
DISPLAY_TEST_SAMPLE = False


# Graph
conv_layers = []
# filter_size, num_filters, pool_size (= 0 if no pull)
conv_layers.append(utils.conv_layer(3, 32, 0))
# conv_layers.append(utils.conv_layer(3, 64, 2))
# conv_layers.append(utils.conv_layer(1, 64, 0))
# conv_layers.append(utils.conv_layer(3, 64, 1))
HIDDEN_UNITS_1 = 128
HIDDEN_UNITS_2 = 32
HIDDEN_1_DROPOUT = True
HIDDEN_2_DROPOUT = False


# Learning
BATCH_SIZE = 200
NUM_STEPS = 10
LEARNING_RATE = 0.002
KEEP_PROB_TRAIN = 0.5 # dropout
IS_STEP_TRAINABLE = False
# Select optimizer: https://www.tensorflow.org/api_docs/python/train/
#
# AdagradOptimizer
# AdadeltaOptimizer
# GradientDescentOptimizer
# MomentumOptimizer
# AdamOptimizer
#
optimizer_function = tf.train.AdamOptimizer
IS_STEP_TRAINABLE = True

# Sanity checks
SUMMARY_INTERVAL = min(SUMMARY_INTERVAL, math.floor(NUM_STEPS / 2))
CHECKPOINT_INTERVAL = min(CHECKPOINT_INTERVAL, math.floor(NUM_STEPS / 2))