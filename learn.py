from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import prepare
import display
import numpy as np
import tensorflow.contrib.learn.python.learn as learn
import tensorflow as tf
from sklearn import metrics
"""Trains and Evaluates the MNIST network using a feed dictionary."""
import random
import math

# pylint: disable=missing-docstring
import argparse
import os.path
import sys
import time
import mnist
from settings import *
import csv

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


# Basic model parameters as external flags.
FLAGS = None

def fill_feed_dict(data_set, images_pl, labels_pl, keep_prob_pl, is_train = True):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
    }
    Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
    Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    images_feed, labels_feed = data_set.next_batch()

    if (is_train):
        keep_prob_val = KEEP_PROB_TRAIN
    else:
        keep_prob_val = 1
    
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      keep_prob_pl: float(keep_prob_val),
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder,
                                   keep_prob_pl,
                                  False)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=LEARNING_RATE,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=NUM_STEPS,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=HIDDEN_UNITS_1,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=HIDDEN_UNITS_2,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=BATCH_SIZE,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=LOG_DIR,
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()

print("initialize training...")
assert(tf.gfile.Exists(FLAGS.log_dir))

tf.gfile.DeleteRecursively(FLAGS.log_dir)
tf.gfile.MakeDirs(FLAGS.log_dir)

"""Train MNIST for a number of steps."""
# Get the sets of images and labels for training, validation, and
# test on MNIST.
data_sets = prepare.DataSets()

# Tell TensorFlow that the model will be built into the default Graph.

# Generate placeholders for the images and labels.
images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                 mnist.IMAGE_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(None))
keep_prob_pl = tf.placeholder(tf.float32)

# Build a Graph that computes predictions from the inference model.
logits = mnist.inference(images_placeholder,
                         FLAGS.hidden1,
                         FLAGS.hidden2,
                        keep_prob_pl)

# Add to the Graph the Ops for loss calculation.
loss = mnist.loss(logits, labels_placeholder)

# Add to the Graph the Ops that calculate and apply gradients.
train_op = mnist.training(loss, FLAGS.learning_rate)

# Add the Op to compare the logits to the labels during evaluation.
eval_correct = mnist.evaluation(logits, labels_placeholder)

# Add the Op to compare the logits to the labels during evaluation.
print_predictions = display.print_prediction(logits, labels_placeholder)

# Build the summary Tensor based on the TF collection of Summaries.
summary = tf.summary.merge_all()

# Add the variable initializer Op.
init = tf.global_variables_initializer()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Create a session for running Ops on the Graph.
sess = tf.Session()

# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

# And then after everything is built:

# Run the Op to initialize the variables.
sess.run(init)


print("Start training...")
# Start the training loop.
for step in xrange(FLAGS.max_steps):
    start_time = time.time()

    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    feed_dict = fill_feed_dict(data_sets.train,
                         images_placeholder,
                         labels_placeholder,
                              keep_prob_pl)

    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, loss_value = sess.run([train_op, loss],
                           feed_dict=feed_dict)

    duration = time.time() - start_time

    # Write the summaries and print an overview fairly often.
    if step % SUMMARY_INTERVAL == 0 and PRINT_SUMMARIES:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

    # Save a checkpoint and evaluate the model periodically.
    if PRINT_SUMMARIES and ((step + 1) % CHECKPOINT_INTERVAL == 0 or (step + 1) == FLAGS.max_steps or step == 0):
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)

#                 # Evaluate against the validation set.
#                 print('Validation Data Eval:')
#                 do_eval(sess,
#                         eval_correct,
#                         images_placeholder,
#                         labels_placeholder,
#                         data_sets.validation)

        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)


test_data = data_sets.test.data



answers = []
def save_pred(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))

num_examples = len(test_data)

func = print_predictions


images = []
titles = []
for i in range(num_examples):
    row = [test_data[i]]
    feed_dict = {
             images_placeholder : row,
             labels_placeholder : [0],
             keep_prob_pl: 1,
            }
    x = sess.run(func, feed_dict=feed_dict)
    answers.append(np.argmax(x))
    
    if i < 9:
        a = np.reshape(row, (28, 28))
        images.append(a)
        titles.append(np.argmax(x))

if DISPLAY_TEST_SAMPLE:
	display.displayImages(images, titles, 33)

if FILENAME_TIMESTAMP:
	OUTPUT_PATH += str(time.time())

with open(OUTPUT_PATH, "w", newline="\n") as ofile:
    writer = csv.writer(ofile)
    writer.writerow(COLUMN_NAMES)
    for i in range(len(answers)):
        writer.writerow([(i + 1), answers[i]])
        
print("done")