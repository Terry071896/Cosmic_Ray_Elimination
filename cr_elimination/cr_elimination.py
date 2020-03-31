# Author: Terry Cox
# GitHub: https://github.com/Terry071896/Cosmic_Ray_Elimination
# Email: tcox@keck.hawaii.edu, tfcox1703@gmail.com

__author__ = ['Terry Cox']
__version__ = '1.0.1'
__email__ = ['tcox@keck.hawaii.edu', 'tfcox1703@gmail.com']
__github__ = 'https://github.com/Terry071896/Cosmic_Ray_Elimination'

import numpy as np
from astropy.visualization import ZScaleInterval, AsinhStretch,SinhStretch, BaseStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from keras.models import load_model, model_from_json
import matplotlib
from matplotlib import pyplot as plt
from model.create_model import create_model
import sys
sys.setrecursionlimit(10**8)

class Cosmic_Ray_Elimination(object):
    '''
    A class built to use machine learning to eliminate cosmic rays from spectral images.

    ...

    Attributes
    ----------
    model : Keras Model
        the convolutional neural network to remove cosmic rays from a spectrial image.
    new_image_data : numpy.ndarray (2D)
        a NumPy array of the spectral image without cosmic rays.
    scale : numpy.ndarray (2D)
        a scaled version of the original spectral image (if required)
    remove_binary : numpy.ndarray (2D)
        a binary matrix of the what pixel values are removed (1 = removed, 0 = kept)

    Methods
    -------
    remove_cosmic_rays(image_data, zscore = 2, pixels_shift = 256)
        the method that removes the cosmic rays from the spectrial image.
    fill_gaps(point, keepers_binary, scale_binary)
        a recursive function to clean up the values that were removed that shouldn't have been removed and vice versa.
    '''

    def __init__(self):
        try:
            json_file = open('model/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights("model/model.h5")
            self.model = loaded_model
        except:
            print('Model not found.  Training model...')
            self.model = create_model()
            print('Model trained.')

    def remove_cosmic_rays(self, image_data, zscore = 2, pixels_shift = 256):
        '''
        The method that removes the cosmic rays from the spectrial image.

        Parameters
        ----------
        image_data : numpy.ndarray (2D)
            the original or scaled version of the original spectral image needing removal of cosmic rays.
        zscore : int or float, optional (default = 2)
            the number of standard deviations away from classifying a pixel value as 1 (above) or 0 (below).
        pixels_shift : int, optional (default = 256)
            the number pixels the 256 x 256 pixel window will slide from left to right.

        Returns
        -------
        numpy.ndarray (2D)
            a NumPy array of the spectral image without cosmic rays.

        Notes
        -----
        The optional parameters, zscore and pixels_shift, are ment to allow for flexiblility of a given to remove more or less pixel values.

        The closer zscore is to 0, the more pixels looked at for removial and naturally, the further from zero the less pixels (minumally the max pixels will be reviewed).
        The speed of the process will take longer the more pixels are reviewed (maybe a second or 2).  It is recommended that the zscore is intended to find the outliers as cosmic rays tend to be.

        zscore cutoff percentile chart:

        zscore | percentile
        -------|-----------
           0   |    50%
           1   |    84%
           2   |    97.5%

        As for pixels_shift, the closer to 1 the integer gets, the more frequent a pixel will be reviewed as the 256 window will slide by that many pixels.
        For example,
        If pixels_shift = 1 (minimum), then pixel:
            (0,0) reviewed 1  -----> just in the first window (top right corner)
            (0,1) reviewed 2  -----> in first and second windows; windows (0,0) and (0,1)
            (1,1) reviewed 4  -----> in windows (0,0), (0,1), (1,0), and (1,1)
            (0,356) reviewed 256 -----> in windows (0, 100-356)
            (356,356) reviewed 65,526 -----> in windows (100-356, 100-356)
        If pixels_shift = 256 (default and maximum), each pixel reviewed only ONCE.

        The benefit of having a lower pixels_shift could be to allow the model to make another judgement at the pixel in a different spot on the 256 frame.
        The problem with having lower pixels_shift values is the time increase as O(n^2) is very computationally expensive.

        Most the time this will not help the accuracy of the model, so I recommend keeping the default.
        '''
        try:
            image_data = np.array(image_data)
            if image_data.ndim != 2:
                print('Needs to be a 2D array.')
                return None
            elif image_data.dtype != np.dtype('float'):
                print('Needs to be a 2D array of floats not', image_data.dtype)
                return None
            elif np.min(image_data.shape) < 256:
                print('\'image_data\' must have a width and length >= 256: ', image_data.shape)
                return None
        except:
            print('Needs to be a 2D array.')
            return None

        if not isinstance(zscore, (int, float)):
            print('\'zscore\' needs to be type int or float.')
            return None

        if not isinstance(pixels_shift, int):
            print('\'pixels_shift\' needs to be type int.')
            return None

        if pixels_shift > 256:
            print('\'pixels_shift\' can be a max of 256.')
            print('pixels_shift = 256')
            pixels_shift = 256

        ############################### Load and Scale ############################################
        max_pixel = np.max(image_data)
        if max_pixel > 1:
            interval = ZScaleInterval()
            vmin,vmax = interval.get_limits(image_data)
            norm = ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch())
            scale = norm.__call__(image_data).data
        else:
            scale = image_data

        z = np.mean(scale)+np.std(scale)*zscore
        if z > 1:
            z = 1
        scale_binary = scale.copy()
        scale_binary[scale_binary >= z] = 1
        scale_binary[scale_binary < z] = 0

        ############################### Build X ############################################

        x1 = np.array(range(0,int(np.ceil(scale.shape[0]/pixels_shift))))*pixels_shift
        x2 = np.array(range(0,int(np.ceil(scale.shape[1]/pixels_shift))))*pixels_shift
        sixteenth = np.round(np.array(range(0,4))*64)

        scale_shape = scale.shape
        X = []
        image256pos = []
        images_orig256 = []
        for i in x1:
            for j in x2:
                if i+256 > scale_shape[0]-1 and j+256 > scale_shape[1]-1:
                    image256 = scale_binary[(scale_shape[0]-256):(scale_shape[0]), (scale_shape[1]-256):(scale_shape[1])]
                    image256pos.append((scale_shape[0]-256, scale_shape[1]-256))
                elif i+256 > scale_shape[0]-1:
                    image256 = scale_binary[(scale_shape[0]-256):(scale_shape[0]), j:(j+256)]
                    image256pos.append((scale_shape[0]-256, j))
                elif j+256 > scale_shape[1]-1:
                    image256 = scale_binary[i:(i+256), (scale_shape[1]-256):(scale_shape[1])]
                    image256pos.append((i, scale_shape[1]-256))
                else:
                    image256 = scale_binary[i:(256+i), j:(256+j)]
                    image256pos.append((i,j))
                images_orig256.append(image256)
                for z in sixteenth:
                    for k in sixteenth:
                        x = image256[z:(z+64), k:(k+64)].reshape(64,64,1)
                        X.append(x)

        ############################### Predictions ############################################
        X = np.array(X)
        Y_pred = self.model.predict(X)
        counter = 0

        keepers = np.zeros(scale.shape)
        counter_matrix = np.zeros(scale.shape)
        for i, point in enumerate(image256pos):
            pred_data_y = np.zeros((256,256))

            for j in sixteenth:
                for k in sixteenth:
                    pred_data_y[j:(j+64), k:(k+64)] = Y_pred[counter].reshape(64,64).round()
                    counter += 1

            keepers[point[0]:(point[0]+256), point[1]:(point[1]+256)] = keepers[point[0]:(point[0]+256), point[1]:(point[1]+256)] + pred_data_y
            counter_matrix[point[0]:(point[0]+256), point[1]:(point[1]+256)] += 1
        keepers = keepers/counter_matrix
        keepers_binary = np.ceil(keepers/np.max(keepers))

        ############################### Fill In Gaps ############################################
        points = zip(*np.where(keepers_binary == 1))

        for point in points:
            keepers_binary = self.fill_space(point, keepers_binary, scale_binary)

        ############################### Remove and Finish ############################################
        remove_binary = scale_binary - keepers_binary
        new_image_data = scale  - scale*remove_binary

        self.new_image_data = new_image_data
        self.remove_binary = remove_binary
        self.scale = scale
        return new_image_data


    def fill_space(self, point, keepers_binary, scale_binary):
        '''
        A recursive function to clean up the values that were removed that shouldn't have been removed and vice versa.

        Parameters
        ----------
        point : tuple or list
            a kept point at which needs to check if the proximity points need to be turned on, off, or nothing.
        keepers_binary : numpy.ndarray (2D)
            a matrix of the predicted pixels that are ment to be kept. (Binary matrix)
        scale_binary : numpy.ndarray (2D)
            a matrix of the true pixels of the original image. A binary matrix where if above the zscore cutoff than 1 else 0.

        Returns
        -------
        numpy.ndarray (2D)
            this is the keepers_binary matrix with the adjusted pixel values that it touches (3 x 3 grid)

        Notes
        -----
        This method is a recursive function that checks the points around a point to see if:
            1. Is this point a False Negative (keepers_binary = 0 and scale_binary = 1)
                Yes - Set to 1, then do the same with the points around this point.
                No - Move on to next check
            2. Is this point a False Positive (keepers_binary = 1 and scale_binary = 0)
                Yes - Set to 0
                No - Nothing
            3. Return new keepers_binary matrix
        '''
        try:
            if keepers_binary.shape != scale_binary.shape:
                print('keepers_binary and scale_binary not the same size')
                return None
        except:
            print('keepers_binary and scale_binary need to be same size 2D numpy arrays.')
            return None

        for i in [point[0]-1,point[0],point[0]+1]:
            for j in [point[1]-1,point[1],point[1]+1]:
                try:
                    if keepers_binary[i,j] == 0 and scale_binary[i,j] == 1 and (i,j) != point:
                        keepers_binary[i,j] = 1
                        return self.fill_space((i,j), keepers_binary, scale_binary)
                    elif keepers_binary[i,j] == 1 and scale_binary[i,j] == 0:
                        keepers_binary[i,j] = 0
                        return keepers_binary
                except:
                    #print('(%s, %s) is out of bounds'%(i,j))
                    continue
        return keepers_binary
