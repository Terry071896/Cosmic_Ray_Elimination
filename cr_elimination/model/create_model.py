from numpy.random import seed
seed(2424)
import tensorflow as tf
tf.random.set_seed(2424)


# Author: Terry Cox
# GitHub: https://github.com/Terry071896/Cosmic_Ray_Elimination
# Email: tcox@keck.hawaii.edu, tfcox1703@gmail.com

__author__ = ['Terry Cox']
__version__ = '1.0.1'
__email__ = ['tcox@keck.hawaii.edu', 'tfcox1703@gmail.com']
__github__ = 'https://github.com/Terry071896/Cosmic_Ray_Elimination'

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras import layers
import numpy as np
from PIL import Image
from zipfile import ZipFile
import keras.backend as K
import sys
import io
import timeit

def custom_loss_function(y_actual, y_predicted):
    '''
    Custom loss function for training the CNN.

    Parameters
    ----------
    y_actual : list
        the actual values (y)
    y_predicted : list
        the predicted values (yhat)

    Returns
    -------
    float
        the loss/error of the 2 lists.

    Notes
    -----
    The metric is ment to help reduce False Negatives.  To do so, the MSE is scaled by:
    (1 - True Positive / n_all)*(1 + False Negative / n_negative).

    Table of importance:

    y_actual | y_predicted
    ---------|-------------
        0    |      0      ----> Good (Most Common)
        0    |      1      ----> Not often bad and easy to fix (Very Uncommon)
        1    |      0      ----> Must Limit (Common- around 40-50% of y_actual = 1)
        1    |      1      ----> Very Good (Common- around 50-60% of y_actual = 1)

    Reducing the False Negatives are more important than reducing the False Positives because the model tends to remove more of features (removing parts of good and all of bad).
    With that being said, it is much less time consuming to fix a False Positive.
    Because the model is built to remove pixel values (remove cosmic rays), it does not happen often for the model to "turn on" a pixel that was "off" before and when it does it is very rare that it is a cosmic ray.

    This allows us to make extra efforts to reduce False Negatives knowing that False Positives are hardly a problem.
    '''

    n11 = K.sum(K.round(y_actual) * K.round(y_predicted)) # find the number of True Positive
    nboth = K.sum(K.round(y_actual)) # find the total actual positives
    n10 = nboth - n11 # find the total number of False Negatives
    b = K.round(y_actual)+100 # create a list where everything has a value NOT 0
    nall = K.sum(b/b) # find the length of the list

    mse = K.mean(K.square(y_actual-y_predicted)) # find the mean squared error
    custom_loss_value = mse*(1-n11/nall)*(1 + K.sum(n10)/nboth) # find loss
    return custom_loss_value # return loss

def create_model():
    '''
    Builds, trains, and stores the CNN model.

    Returns
    -------
    Keras Model
        the convolutional neural network to remove cosmic rays from a spectrial image.

    Notes
    -----
    The structure of the model is as such, taking in a (64, 64, 1) image.

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_1 (Conv2D)            (None, 60, 60, 32)        832
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 20, 20, 32)        0
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 16, 16, 64)        51264
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 4096)              0
    _________________________________________________________________
    dense_1 (Dense)              (None, 6000)              24582000
    _________________________________________________________________
    dense_2 (Dense)              (None, 4096)              24580096
    =================================================================
    Total params: 49,214,192
    Trainable params: 49,214,192
    Non-trainable params: 0
    _________________________________________________________________
    '''

    Y = [] # init Y as list
    X_full = [] # init X_full as list
    sixteenth = np.round(np.array(range(0,4))*64) # create list of index points for top left corner of images to break a 256x256 into 16, 64x64 images.

    filename = 'Original.zip' # init filename

    start_time = timeit.default_timer() # start time

    archive0 = ZipFile('model/Original.zip', 'r') # make Original.zip archive readable
    archive1 = ZipFile('model/Solutions.zip', 'r') # make Solutions.zip archive readable

    for i in range(0,780): # loop through all pictures
        fileX = 'Original/image_actual_%s.png'%(i) # file name of picture in folder
        image_data = archive0.read(fileX) # read picture
        fh = io.BytesIO(image_data) # get bytes
        img = Image.open(fh).convert('L') # convert to Image object with data of image in floats

        data_x = np.round(np.array(img.getdata())/255).reshape(256,256) # make image data 0-1 in 256x256 2D array

        fileY = 'Solutions/image_actual_%s.jpeg'%(i) # file name of picture in folder
        image_data = archive1.read(fileY) # read picture
        fh = io.BytesIO(image_data) # get bytes
        img = Image.open(fh).convert('L') # convert to Image object with data of image in floats

        data_y = np.round(np.array(img.getdata())/255).reshape(256,256) # make image data 0-1 in 256x256 2D array


        for j in sixteenth: # top index of image
            for k in sixteenth: # far left index of image
                y = data_y[j:(j+64), k:(k+64)].reshape(1,-1)[0] # store 1096x1 numpy array of Solutuion image
                x = data_x[j:(j+64), k:(k+64)] # store 64x64 numpy 2D array of Original image
                X_full.append(x) # append x to X_full list
                Y.append(y) # append y to Y list

    X_full = np.array(X_full).reshape((len(X_full), 64,64,1)) # make X_full numpy array and reshape so X_full has (len(X_full),64,64,1) shape
    Y = np.array(Y) # make Y a numpy array
    elapsed = timeit.default_timer() - start_time # stop time
    print(elapsed) # print time

    # define the keras model
    model = Sequential() # Model 2: look at comment and adjust

    model.add(layers.Conv2D(32,(5,5),activation='relu', input_shape=(64,64,1))) # input shape of (64,64,1) and 1st convolution layer of 32 nodes and filter of (5,5)     Model 2: 16 (5,5)
    model.add(layers.MaxPooling2D((3, 3))) # max pooling layer of (3,3) filter      Model 2: Max (2,2)
    model.add(layers.Conv2D(64, (5, 5), activation='relu')) # 2nd convolution layer of 64 nodes and filter of (5,5)      Model 2: 86 (5,5)
    model.add(layers.MaxPooling2D((2, 2))) # max pooling layer of (2,2)       Model 2: Max (2,2)
    # model.add(layers.Conv2D(103, (5, 5), activation='relu')) # Model 2: uncomment
    # model.add(layers.MaxPooling2D((3, 3))) # Model 2: uncomment
    model.add(layers.Flatten()) # flatten to dense layer of 1096 nodes

    model.add(Dense(6000, activation='relu')) # 1st dense layer of 6000 nodes      Model 2: 5100
    model.add(Dense(len(Y[0]), activation='relu')) # output layer of 1096 nodes

    model.compile(loss=custom_loss_function, optimizer='adam', metrics=['mse','mae']) # compile function with loss as custom_loss_function and optimizer as adam
    history = model.fit(X_full, Y, epochs=20, verbose = 1) # fit the model

    model_json = model.to_json() # make model json
    with open("model/model.json", "w") as json_file: # open model.json as writable
        json_file.write(model_json) # write structure to model.json
    model.save_weights("model/model.h5") # save weights of model as model.h5

    return model
