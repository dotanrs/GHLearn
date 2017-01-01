
# GHLearner

The GlitzHat Learner library creates a TensorFlow neural network for classifying square images.
The network has zero or more convolution layers and two fully connceted layers.

The number of convolution layers and their properties can be defined in the settings file.

The size of the fully connceted layers can also be defined, and a dropout can be introduce in
one or both of them.

A number of other properties can be defined such as the type of optimizer, number of steps etc.

The library works with tensorboard to view the network and it's learning process.

## Input assumptions

all data is in square images, size defined in settings

the rightmost column is the labels. The first row is the headers

## Requirements:

numpy, tensorflow, PIL

# Usage

1. Update settings.py
2. `$ <path>: make`

to view tensorboard:
`$ tensorboard --logdir /tmp/tensorflow/mnist/logs/fully_connected_feed`
(in case changed LOG_DIR in settings, switch the url)

If port is taken, add 
`--port 6007` (or 6008 etc)