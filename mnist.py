from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import numpy as np
import settings

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = settings.NUM_CLASSES

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = settings.IMAGE_SIZE
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

DEFAULT_WEIGHT_STD = 1.0 / math.sqrt(float(IMAGE_PIXELS))

def variable_with_wights(shape, name="weights"):

  return tf.Variable(
            tf.truncated_normal(shape,
                          stddev=DEFAULT_WEIGHT_STD),
            name=name)

def zeroed_variable(shape, name="biases"):

  return tf.Variable(tf.zeros(shape),
                         name=name)

def conv_2d(image, weights):
  return tf.nn.conv2d(image, weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')

def max_pool(conv, num):
  return tf.nn.max_pool(conv,
                        ksize=[1, num, num, 1],
                        strides=[1, num, num, 1],
                        padding='SAME')

def calc_pixels_for_conv(in_pixels, padding, filter_size, pool_step = 1):
  
  # NOTE: assumptions
  #      - pool = pool_step
  #      - stride = 1 in filter 
  pool = pool_step

  #return math.ceil((((in_pixels + padding) - filter_size + 1) + padding) / pool)

  # conv
  # output is the same size as input
  conv_with_padding = in_pixels - 1 + filter_size

  # pool
  out_size = math.ceil( float(in_pixels) / float(pool_step))
  pool_with_padding = (out_size - 1) * pool - 1 

  # return in_pixels
  # print("PIXEL CALC pix:%d, filt:%d, pool:%d, out:%d" %
  #         (in_pixels, filter_size, pool_step, out_size))

  return out_size # math.ceil((((pixels + padding) - filter_size + 1) + padding) / pool_step)


def inference(images, hidden1_units, hidden2_units, keep_prob):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """

  conv_layers = settings.conv_layers

  padding = 2 # doesn't matter
  pixels = IMAGE_SIZE
  dimensions = 1

  layer_inputs = {}
  reshaped_image = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
  layer_inputs[0] = reshaped_image
  count_layers = 0
  previous_number_filters = 1

  for index, properties in enumerate(conv_layers):
    with tf.name_scope('conv_layer_' + str(index)):
      weights = variable_with_wights([
          properties.filter_size,
          properties.filter_size,
          previous_number_filters,
          properties.num_filters
        ])

      biases  = zeroed_variable([properties.num_filters])

      if properties.pool_size:
        conv_layers[index] = tf.nn.relu(conv_2d(layer_inputs[index], weights) + biases)
        layer_inputs[index + 1] = max_pool(conv_layers[index], properties.pool_size)
        pixels = calc_pixels_for_conv(pixels,
                                      padding, 
                                      properties.filter_size, 
                                      properties.pool_size
                                      )
        
      else:
        layer_inputs[index + 1] = tf.nn.relu(conv_2d(layer_inputs[index], weights) + biases)


    count_layers = index + 1
    previous_number_filters = properties.num_filters
    dimensions = properties.num_filters

  # Hidden 1
  with tf.name_scope('hidden1'):

    total_pixels = pixels * pixels * dimensions

    weights = variable_with_wights([
                  total_pixels,
                  hidden1_units
                  ])
    biases  = zeroed_variable([hidden1_units])
    image_vector = tf.reshape(layer_inputs[count_layers], [
                      -1,
                      total_pixels
                      ])

    hidden1 = tf.nn.relu(tf.matmul(image_vector, weights) + biases)

    # Note: tensorflow automatically scales this
    if settings.HIDDEN_1_DROPOUT:
      h1_dropout = tf.nn.dropout(hidden1, keep_prob)


  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = variable_with_wights([hidden1_units, hidden2_units])
    biases  = zeroed_variable([hidden2_units])

    if settings.HIDDEN_1_DROPOUT:
      hidden2 = tf.nn.relu(tf.matmul(h1_dropout, weights) + biases)
    else:
      hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    if settings.HIDDEN_2_DROPOUT:
      h2_dropout = tf.nn.dropout(hidden2, keep_prob)

  # Linear
  with tf.name_scope('softmax_linear'):
    weights = variable_with_wights([hidden2_units, NUM_CLASSES])
    biases  = zeroed_variable([NUM_CLASSES])

    if settings.HIDDEN_2_DROPOUT:
      logits = tf.matmul(h2_dropout, weights) + biases
    else:
      logits = tf.matmul(hidden2, weights) + biases

  return logits


def loss(logits, labels):
  labels = tf.to_int64(labels)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)

  optimizer = settings.optimizer_function(learning_rate)

  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=settings.IS_STEP_TRAINABLE)

  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.

  return tf.reduce_sum(tf.cast(correct, tf.int32))