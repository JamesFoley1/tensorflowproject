from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K

# What is a tensor? 
# tensor is an algebraic object that describes a 
# linear mapping from one set of algebraic objects to another. 

# Conv2D fucntion - convolution kernel that is convolved
# with the layer input to produce a tensor of outputs.

class LeNet:
    @staticmethod
    def build(numChannels,
        imgRows,
        imgCols,
        numClasses,
        activation='relu',
        weightsPath=None):

        # relu = Rectified Linear Unit
        # Piecewise linear function that returns the input
        # if positive otherwise, returns 0

        # initialize model. First layer needs to be a shape
        model = Sequential()
        inputShape = (imgRows, imgCols, numChannels)

########### IF CHECK FOR CHANNEL TYPE, SKIP #################
        #if we use channels first, update input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)
#############################################################
    
        # define set of CONV => ACTIVATION => POOL LAYERS 
        model.add(Conv2D(20, 5, padding="same", #20 convolution filters with size of 5x5
            input_shape=inputShape))

        # ReLU activation
        model.add(Activation(activation))

        # 2x2 max-pooling kernel in both x and y directions

        # Kernel stride of 2 - think of sliding window that slides across the activation volume,
        # taking max operation of each region, while taking a step of 2 pixels in horizontal and veritcal directions
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # adding 50 convolution filters. common to increase CONV filters in deeper layers
        model.add(Conv2D(50, 5, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # now to fully connect layers (called "dense" layers)

        # Flatten takes output of previous maxpooling2d and condenses it into a single vector,
        # which allows us to apply dense (fully connected) layers.
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        # VERY IMPORTANT TO INCLUDE
        # second dense layer accepts variable size which defines the
        # number of class labels represented by classes variable
        # MNIST dataset contains 10 classes (1 for each digit)
        model.add(Dense(numClasses))

        # softmax classifier - multinomial logistic regression
        # returns a list of probabilities for each of the 10 class labels.
        # largest is chosen as the final classification from the network
        model.add(Activation("softmax"))

        # if supplied a weights path, load the weights
        # this means the model was pre-trained
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # returns constructed network architecture
        return model
