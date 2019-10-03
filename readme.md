# Basic Convolutional Neural Network utilizing the mnist data to identify 0-9 handwritten numbers


## NOTES
cnn sub-module stores our convolutional neural network implementations and helper utilities related to CNNS

networks sub-module is where the network implementations will be stored

lenet.py defines class LeNet, which implements Python + Keras

lenet_mnist.py is the driver program which instantiates leNet network architecture, trains the model, and evaluates network performance

UNTRAINED CMD - python lenet_mnist.py --save-model 1 -l
PRE-TRAINED CMD - python lenet_mnist.py --load-model 1 --weights my_model.hdf5