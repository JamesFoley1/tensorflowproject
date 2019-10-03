from pyimagesearch.cnn.networks.lenet import LeNet
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2

### construct argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int,
    help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int,
    help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
    help="(optional) path to weights file")
args = vars(ap.parse_args())



print("\n downloading MNIST Data...\n")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

#################################  FRAMEWORK DEPENDENT  ##############################################
#################### Change image shape based on framework used ######################################

# if channel first ordering, reshape
if K.image_data_format() == "channels_first":
    trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))

# otherwise, channels last
else:
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

########################################################################################################
########################################################################################################



# scale data to range of [0, 1] in order to treat all images in the same manner and for learning rate
# for further reasons/resources: https://www.linkedin.com/pulse/keras-image-preprocessing-scaling-pixels-training-adwin-jahn/
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

# transform training/testing labels into vectors - [0, classes] range
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialize optimizer + model
print("compiling model \n")


########################################################################################################
#####################################  COMPILATION  ####################################################

# Stochastic Gradient Descent (SGD) - optimizer which takes in learning rate value
opt = SGD(lr = 0.01)
model = LeNet.build(numChannels = 1, imgRows = 28, imgCols = 28, numClasses = 10,
    weightsPath = args["weights"] if args["load_model"] != "" else None)


# Apply categorical cross-entropy loss function - converts labels from integers to a vector
# which ranges from [0, classes] (0-10). correct = 1, incorrect = 0

# Categoical cross-entropy gives the probability that an image
# of a number is for example, a 4 or a 9.
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

# OUTPUT: so with 10 labels, each label represents a 10-dimensional vector.
# label #3 should now look like [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]




######################### IF WE DON'T PRE-TRAIN #############################
if args["load_model"] != "":
    print("training...\n")
    model.fit(trainData, trainLabels, batch_size = 128, epochs = 10, verbose = 1)

    print("evaluating... \n")
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size = 128, verbose = 1)
    print("accuracy: {:.2f}%".format(accuracy * 100))
##############################################################################

######################## IF WE WANT TO SAVE WEIGHTS #########################
if args["save_model"] is not None and args["save_model"] > 0:
    print("dumping weights to file...\n")
    # model.save_weights("", overwrite = True)
    model.save('my_model.hdf5')
#############################################################################


################## FOR RANDOM CHOICE FROM TEST CASES ########################

for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    
    # classify probability digit
    probs = model.predict(testData[np.newaxis, i])

    # prediction is obtained by finding index of class label with the largest probability
    prediction = probs.argmax(axis = 1)


############### NOW WE GIVE OURSELVES SOME VISUAL FEEDBACK ##################

    # extra image from testData for channels first, then order
    if K.image_data_format() == "channels_first":
        image = (testData[i][0] * 255).astype("uint8")

    # otherwise channels_last
    else:
        image = (testData[i] * 255).astype("uint8")

    # merge channels into one image
    image = cv2.merge([image] * 3)

    # resize from 28x28 so we can see it
    image = cv2.resize(image, (96, 96), interpolation = cv2.INTER_LINEAR)

    # show the image and prediction with hershey simplex font
    cv2.putText(image, str(prediction[0]), (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    print("Predicted: {}, Actual: {}".format(prediction[0], np.argmax(testLabels[i])))

    cv2.imshow("Digit", image)
    cv2.waitKey(0)
