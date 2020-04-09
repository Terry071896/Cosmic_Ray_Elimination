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
import timeit
from sklearn.cluster import KMeans
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
    times : list
        a list of how long each section of the remove_cosmic_rays method took.

    Methods
    -------
    remove_cosmic_rays(image_data, zscore = 2, pixels_shift = 256, box_width = 7, box_height = 5, max_dim = 2)
        the method that removes the cosmic rays from the spectrial image.
    fill_gaps(point, keepers_binary, scale_binary)
        a recursive function to clean up the values that were removed that shouldn't have been removed and vice versa.
    '''

    def __init__(self, model_name = 'model'):
        try:
            json_file = open('model/%s.json'%model_name, 'r') # open model structure
            loaded_model_json = json_file.read() # read json
            json_file.close() # close file
            loaded_model = model_from_json(loaded_model_json) # make json keras model
            loaded_model.load_weights("model/%s.h5"%model_name) # import the weights
            self.model = loaded_model # store loaded model as self.model
        except:
            if model_name == 'model':
                print('Model not found.  Training model...') # if model is not found, then build model
                self.model = create_model() # build model
                print('Model trained.')
            else:
                json_file = open('model/%s.json'%model_name, 'r') # open model structure
                loaded_model.load_weights("model/%s.h5"%model_name) # import the weights

    def remove_cosmic_rays(self, image_data, zscore = 2, pixels_shift = 256, box_width=7, box_height=5, max_dim=2):
        '''
        The method that removes the cosmic rays from the spectrial image.

        Parameters
        ----------
        image_data : numpy.ndarray (2D)
            the original or scaled version of the original spectral image needing removal of cosmic rays.
        zscore : int/float or 'kmeans', optional (default = 2)
            the number of standard deviations away from classifying a pixel value as 1 (above) or 0 (below).
            if 'kmeans', then the optimal zscore will be found using the unsupervised machine learning algorithm k-means.
        pixels_shift : int, optional (default = 256)
            the number pixels the 256 x 256 pixel window will slide from left to right.
        box_width : int, optional (default = 7)
            the width of the window around a point to be replaced by the median pixel value of the box.
        box_height : int, optional (default = 5)
            the height of the window around a point to be replaced by the median pixel value of the box.
        max_dim : int or None, optional (default = 2)
            the maximum dimentions allowed for touching removed pixels that existed before to be kept instead.

        Returns
        -------
        numpy.ndarray (2D)
            a NumPy array of the spectral image without cosmic rays.

        Notes
        -----
        The optional parameters, zscore, pixels_shift, box_width, box_height, and max_dim, are ment to allow for flexiblility of a given to remove more or less pixel values.

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

        Parameters box_width and box_height are the dimentions used to create a window around a pixel fixed to be removed.
        With the point in the middle of the box, the pixel value is to be replaced by the median pixel value in the box.
        Not only is the point of this box ment to replace the pixel with a reasonable pixel based off of the proximity pixels, but is very import in saving a pixel on a spectral line that is ment to be removed.
        So it is recommended to have a window that would best represent the shape of the average spectral line, but not large enough to lose the beneifit of the proximity replacing the value.

        The last parameter max_dim is the maximum dimention for the recursive function "fill_gaps" which cleans up the values that are to be removed that shouldn't be removed and vice versa.
        This value should be low (0-2) if there are many spectral lines and lots of cosmic rays.
        If the value is None, then there is no maximum and any kept pixel that is touching a pixel that is turned off and the original is turned on will turn it back on.
        What needs to be watched out for are cosmic rays that are touching the spectrial lines.
        If the value is None, it will fill in all the cosmic rays that are touching any reviewed spectral lines.
        With that being said, if there are limited lines, the having a very high number or setting to None is advised as it is quicker and more efficient than relying on fixing the issues with the median of the pixel box made by box_width and box_height.
        '''

        ###################### Check Parameters ###############################

        try:
            image_data = np.array(image_data) # make image data numpy array
            if image_data.ndim != 2: # needs to be 2D
                print('Needs to be a 2D array.')
                return None
            elif image_data.dtype != np.dtype('float'): # check to see if not array of floats
                try:
                    interval = ZScaleInterval() # init zscale object
                    vmin,vmax = interval.get_limits(image_data) # find interval limits
                    norm = ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch()) # init zscale with a sinh stretch
                    image_data = norm.__call__(image_data).data # scale and redefine image_data
                except:
                    print('Needs to be a 2D array of floats not', image_data.dtype) # if not int or float print error
                    return None
            elif np.min(image_data.shape) < 256: # make sure input has a shape greater than 256
                print('\'image_data\' must have a width and length >= 256: ', image_data.shape)
                return None
        except:
            print('Needs to be a 2D array.') # if not array print error
            return None

        if not isinstance(zscore, (int, float)): # zscore must be int or float or 'kmeans'
            if zscore != 'kmeans': # because not int or float then check to see if 'kmeans'
                print('\'zscore\' needs to be type int or float or \'optimal\'.') # print error and return
                return None

        if not isinstance(pixels_shift, int): # pixels_shift must be an int
            print('\'pixels_shift\' needs to be type int.') # print error and return
            return None

        if pixels_shift > 256: # pixels_shift cannot be greater 256
            print('\'pixels_shift\' can be a max of 256.')
            print('pixels_shift = 256')
            pixels_shift = 256 # redefine pixels_shift to be max at 256

        if not isinstance(box_width, int): # box_width must be int
            print('\'box_width\' must be an int.')
            return None

        if not isinstance(box_height, int): # box_height must be int
            print('\'box_height\' must be an int')
            return None

        if not max_dim is None and not isinstance(max_dim, int): # max_dim must be None or int
            print('\'max_dim\' must be either int or None.')
            return None

        times = [] # init times as list
        total_time = 0 # init total_time
        start_time = timeit.default_timer() # start timer
        ############################### Load and Scale ############################################
        max_pixel = np.max(image_data) # store max pixel of image_data
        if max_pixel > 1: # if max_pixel is > 1 then image_data not scaled
            interval = ZScaleInterval() # init zscale object
            vmin,vmax = interval.get_limits(image_data) # find interval limits
            norm = ImageNormalize(vmin=vmin,vmax=vmax,stretch=SinhStretch()) # init zscale with a sinh stretch
            scale = norm.__call__(image_data).data # set scale to the scaled image_data
        else:
            scale = image_data # becuase max_pixel < 1, then must be scaled. Rename image_data "scale"


        if zscore == 'kmeans': # if True, then use k-means method to find optimal pixels to look at (the bright group specifically)
            X_kmeans = scale.reshape(-1,1) # make X_scale with each pixel in its own list... [[0.34], [0.76], ...]
            kmeans = KMeans(n_clusters=2, random_state=0).fit(X_kmeans) # preform k-means algorithm to produce to groups
            scale_binary = kmeans.labels_.reshape(scale.shape) # reshape groups 0 and 1 back into shape of scale and call scale_binary.  1 = possible cosmic rays and 0 = ignore (probably not cosmic rays)
            if bool(kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]): # make sure that group 1 is the brightest group
                scale_binary = -1*(scale_binary-1) # make 0 ---> 1 and 1 ---> 0
            x = scale_binary.reshape(1,-1)[0]*(scale.reshape(1,-1)[0]) # list of bright groups pixel values if not 0
            print('zscore = %s'%((np.min(x[x>0])-np.mean(scale))/np.std(scale))) # find and print the zscore (the minimum of bright group is the line break then zscore = (value - mean)/standard deviation)
        else:
            z = np.mean(scale)+np.std(scale)*zscore # find pixel value to seperate
            if z > 1: # if z is greater than max pixel value make max pixel value a.k.a. 1
                z = 1
            scale_binary = scale.copy() # make copy of scale called scale_binary
            scale_binary[scale_binary >= z] = 1 # if pixel value in scale_binary >= z make 1
            scale_binary[scale_binary < z] = 0 # if pixel value in scale_binary < z make 0

        elapsed = timeit.default_timer() - start_time # stop timer
        total_time += elapsed # add time to total_time
        times.append('Import and Scale: %s'%elapsed) # add time to times list
        start_time = timeit.default_timer() # start timer
        ############################### Build X ############################################

        x1 = np.array(range(0,int(np.ceil(scale.shape[0]/pixels_shift))))*pixels_shift # make list of top pixels for each window
        x2 = np.array(range(0,int(np.ceil(scale.shape[1]/pixels_shift))))*pixels_shift # make list of far left pixels for each window
        sixteenth = np.round(np.array(range(0,4))*64) # make list of top left corner to break 256 pix image to 16 different 64 pix images

        scale_shape = scale.shape # store scale shape
        X = [] # init X as list
        image256pos = [] # init image256pos as list
        images_orig256 = [] # init images_orig256 as list
        for i in x1: # top pixels for windows
            for j in x2: # far left pixels for windows
                if i+256 > scale_shape[0]-1 and j+256 > scale_shape[1]-1: # if window is out of bounds in both vertical and horizontal direction (a.k.a bottom right corner)
                    image256 = scale_binary[(scale_shape[0]-256):(scale_shape[0]), (scale_shape[1]-256):(scale_shape[1])] # reshift window to catch bottom right corner of image and store 256 windowed image
                    image256pos.append((scale_shape[0]-256, scale_shape[1]-256)) # add top left corner of this window
                elif i+256 > scale_shape[0]-1: # if window is out of bounds in the vertical direction (a.k.a bottom of image)
                    image256 = scale_binary[(scale_shape[0]-256):(scale_shape[0]), j:(j+256)] # reshift window to catch bottom row of image and store 256 windowed image
                    image256pos.append((scale_shape[0]-256, j)) # add top left corner of this window
                elif j+256 > scale_shape[1]-1: # if window is out of bounds in the horizontal direction (a.k.a far right side of image)
                    image256 = scale_binary[i:(i+256), (scale_shape[1]-256):(scale_shape[1])] # reshift window to catch the far right row of image and store 256 windowed image
                    image256pos.append((i, scale_shape[1]-256)) # add top left corner of this window
                else: # not out of bounds anywhere
                    image256 = scale_binary[i:(256+i), j:(256+j)] # no need to reshift, so just store 256 windowed image
                    image256pos.append((i,j)) # add top left corner of this window
                images_orig256.append(image256) # store 256 windowed image
                for z in sixteenth: # top pixel for second window of 256 image
                    for k in sixteenth: # far left pixel for second window of 256 image
                        x = image256[z:(z+64), k:(k+64)].reshape(64,64,1) # store 64,64,1 images as x
                        X.append(x) # add x to X

        elapsed = timeit.default_timer() - start_time # stop timer
        total_time += elapsed # add time to total_time
        times.append('Build X: %s'%elapsed) # add time to timers
        start_time = timeit.default_timer() # start timer
        ############################### Predictions ############################################
        X = np.array(X) # make X a numpy array
        Y_pred = self.model.predict(X) # use CNN model to predict X
        counter = 0 # init counter as int

        keepers = np.zeros(scale.shape) # init matrix of 0 same size as scale
        counter_matrix = np.zeros(scale.shape) # init matrix of 0 same size as scale
        for i, point in enumerate(image256pos): # loop through top left corner points of windowed images
            pred_data_y = np.zeros((256,256)) # init 256,256 matrix of 0's

            for j in sixteenth: # top pixel for second window of 256 image
                for k in sixteenth: # far left pixel for second window of 256 image
                    pred_data_y[j:(j+64), k:(k+64)] = Y_pred[counter].reshape(64,64).round() # take the 16 predicted pieces of the 256 image and reconstruct predicted 256 image
                    counter += 1

            keepers[point[0]:(point[0]+256), point[1]:(point[1]+256)] = keepers[point[0]:(point[0]+256), point[1]:(point[1]+256)] + pred_data_y # add 256 image in its correct spot in matrix keepers
            counter_matrix[point[0]:(point[0]+256), point[1]:(point[1]+256)] += 1 # add 1 to the window of the counter_matrix
        keepers = keepers/counter_matrix # average the pixels in entire keepers matrix
        keepers_binary = np.ceil(keepers/np.max(keepers)) # make keepers_binary from keepers (if value non 0 then it is 1)

        elapsed = timeit.default_timer() - start_time # stop timer
        total_time += elapsed # add time to total_time
        times.append('Predictions: %s'%elapsed) # add time to timers
        start_time = timeit.default_timer() # start time
        ############################### Fill In Gaps ############################################
        if max_dim is None or max_dim > 0: # Check if fill in gaps section is needed (if max_dim is 0 no need for this method)
            points = list(zip(*np.where(keepers_binary == 1))) # create list of tuples of points where keepers_binary values are 1

            for point in points: # go over each keeper point
                keepers_binary = self.fill_space(point, keepers_binary, scale_binary, max_dim=max_dim) # recursive function to fill in spaces that should be filled in around keepers that were eliminated.


        remove_binary = scale_binary - keepers_binary # make matrix of items that need to be removed (if 1 then set to be removed)

        if box_width > 1 or box_height > 1: # make sure that there is a box bigger than 1x1
            new_image_data = scale.copy() # make copy of scale as new_image_data
            points = list(zip(*np.where(remove_binary == 1))) # create list of tuples of points where remove_binary values are 1
            imin_shift = int(np.floor(box_height/2)) # how far down box should be around pixel
            imax_shift = int(np.ceil(box_height/2)) # how far up box should be around pixel
            jmin_shift = int(np.floor(box_width/2)) # how far left box should be around pixel
            jmax_shift = int(np.ceil(box_width/2)) # how far right box should be around pixel
            for point in points: # go over each point to be removed
                imin = point[0]-imin_shift # find bottom of box
                imax = point[0]+imax_shift # find top of box
                jmin = point[1]-jmin_shift # find left edge of box
                jmax = point[1]+jmax_shift # find right edge of box
                if imin < 0: # check to see if out of bounds top
                    imin = 0 # set to top of image
                if jmin < 0: # check to see if out of bounds left
                    jmin = 0 # set to left edge of image
                try:
                    values = list(new_image_data[imin:imax, jmin:jmax].shape(1,-1)[0]) # take pixels box around point from new_image_data nd store as values and shape to 1D list
                except: # if that fails then box is out of bounds by bottom or right edges
                    values = [] # init values as list
                    for i in range(imin, imax): # loop top to bottom of box
                        for j in range(jmin, jmax): # loop left to right of box
                            try:
                                values.append(new_image_data[i,j]) # if in bounds new_image_data pixel at that point to values
                            except:
                                #print('(%s, %s) is out of bounds'%(i,j))
                                continue # if failed then continue

                sorted = np.sort(values) # sort values
                index = int(np.floor(len(sorted)/2)+1) # find index of middle (odd) to one above middle (even)
                med = sorted[index] # store median
                new_image_data[point] = med # replace pixel value at point with med

        elapsed = timeit.default_timer() - start_time # stop time
        total_time += elapsed # add time to total_time
        times.append('Fill the gaps: %s'%elapsed) # add time to timers
        start_time = timeit.default_timer() # start time
        ############################### Remove and Finish ############################################

        if box_width > 1 or box_height > 1: # make sure that there is a box gigger than 1x1
            diff = scale - new_image_data # find the difference between the original scaled image and the new image
            points = list(zip(*np.where(diff < 0))) # list of points where the new image "turned on" a point that should NOT be turned on (not turned on in original)
            for point in points: # loop through points
                new_image_data[point] = scale[point] # replace mistake pixel with original pixel value

            remove_binary = np.round(scale - new_image_data) # make matrix of items that need to be removed (if 1 then set to be removed and if 0 then kept)

            points = list(zip(*np.where(remove_binary == 1))) # create list of tuples of points where remove_binary values are 1
            imin_shift = int(np.floor(box_height/2)) # how far down box should be around pixel
            imax_shift = int(np.ceil(box_height/2)) # how far up box should be around pixel
            jmin_shift = int(np.floor(box_width/2)) # how far left box should be around pixel
            jmax_shift = int(np.ceil(box_width/2)) # how far right box should be around pixel
            for point in points: # go over each point to be removed
                imin = point[0]-imin_shift # find bottom of box
                imax = point[0]+imax_shift # find top of box
                jmin = point[1]-jmin_shift # find left edge of box
                jmax = point[1]+jmax_shift # find right edge of box
                if imin < 0: # check to see if out of bounds top
                    imin = 0 # set to top of image
                if jmin < 0: # check to see if out of bounds left
                    jmin = 0 # set to left edge of image
                try:
                    values = list(new_image_data[imin:imax, jmin:jmax].shape(1,-1)[0]) # take pixels box around point from new_image_data nd store as values and shape to 1D list
                except: # if that fails then box is out of bounds by bottom or right edges
                    values = [] # init values as list
                    for i in range(imin, imax): # loop top to bottom of box
                        for j in range(jmin, jmax): # loop left to right of box
                            try:
                                values.append(new_image_data[i,j]) # if in bounds new_image_data pixel at that point to values
                            except:
                                #print('(%s, %s) is out of bounds'%(i,j))
                                continue # if failed then continue

                new_image_data[point] = np.min(values) # replace pixel value at point with the minimum pixel value in list values
        else:
            new_image_data = scale - remove_binary*scale # if the pixel box is 1x1 or does not make sense, then pixels to be removed will now be 0 in the new image


        elapsed = timeit.default_timer() - start_time # stop time
        total_time += elapsed # add time to total_time
        times.append('Remove and Finish: %s'%elapsed) # add time to timers
        times.append('Total time: %s'%total_time) # start time

        self.times = times # make times an attribute
        self.new_image_data = new_image_data # make new_image_data an attribute
        self.remove_binary = remove_binary # make remove_binary an attribute
        self.scale = scale # make scale an attribute
        return new_image_data # return the new image without cosmic rays.


    def fill_space(self, point, keepers_binary, scale_binary, dim=0, max_dim=2):
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
        dim : int, optional (default = 0)
            number of proximity/touching pixels from original pixel.
        max_dim : int or None, optional (default = 2)
            the maximum dimentions (relivant touching pixels) allowed to iterate from the original pixel. If None, then there is no limit.
        Returns
        -------
        numpy.ndarray (2D)
            this is the keepers_binary matrix with the adjusted pixel values that it touches (3 x 3 grid)

        Notes
        -----
        This method is a recursive function that checks the points around a point to see if:
            1. Is this point a False Negative (keepers_binary = 0 and scale_binary = 1) and dim is less than max_dim
                Yes - Set to 1, then do the same with the points around this point and add 1 to dim.
                No - Move on to next check
            2. Is this point a False Positive (keepers_binary = 1 and scale_binary = 0)
                Yes - Set to 0
                No - Nothing
            3. Return new keepers_binary matrix
        '''
        try:
            if keepers_binary.shape != scale_binary.shape: # make sure keepers_binary and scale_binary are the same shape
                print('keepers_binary and scale_binary not the same size') # print error
                return None
        except:
            print('keepers_binary and scale_binary need to be same size 2D numpy arrays.') # if cannot get a shape from keepers_binary or scale_binary then one or both are not lists.
            return None

        if max_dim is None: # if max_dim is None, then make sure that not limited by dimentions
            dim = 0 # fix dim at 0 so it will always be less than the max_dim
        else:
            dim += 1 # iterate dim so it moves 1 set closer to the max_dim

        for i in [point[0]-1,point[0],point[0]+1]: # check points touching point
            for j in [point[1]-1,point[1],point[1]+1]:
                try: # if fail then out of bounds
                    if keepers_binary[i,j] == 0 and scale_binary[i,j] == 1 and (max_dim is None or dim < max_dim): # see if point is a False Negative and max dimention has not been reached
                        keepers_binary[i,j] = 1 # Set point to keeper (make 1)
                        self.fill_space((i,j), keepers_binary, scale_binary, dim=dim, max_dim = max_dim) # check points around new point (same process)
                    elif keepers_binary[i,j] == 1 and scale_binary[i,j] == 0: # see if point is a False Positive
                        keepers_binary[i,j] = 0 # reset point to its original state of "off"
                except:
                    #print('(%s, %s) is out of bounds'%(i,j))
                    continue # if out of bounds, then continue on without changing anything

        return keepers_binary # return the new keepers_binary
