import numpy as np
# from astropy.io import fits
# from astropy.utils.data import get_pkg_data_filename
from astropy.visualization import ZScaleInterval, AsinhStretch,SinhStretch, BaseStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from keras.models import load_model, model_from_json
import matplotlib
from matplotlib import pyplot as plt
from model.create_model import create_model
import sys
sys.setrecursionlimit(10**8)

class Cosmic_Ray_Elimination(object):
    def __init__(self):
        try:
            json_file = open('model/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("model/model.h5")
            self.model = loaded_model
        except:
            print('Model not found.  Training model...')
            self.model = create_model()
            print('Model trained.')

    def remove_cosmic_rays(self, image_data, zscore = 2, pixels_shift = 256):

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
