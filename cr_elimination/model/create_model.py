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

def custom_loss_function(y_actual, y_predicted):
    n11 = K.sum(K.round(y_actual) * K.round(y_predicted))
    nboth = K.sum(K.round(y_actual))
    n10 = nboth - n11

    mse = K.mean(K.square(y_actual-y_predicted))
    custom_loss_value = mse * (1 + K.sum(n10)/nboth)
    return custom_loss_value

def create_model():

    Y = []
    X_full = []
    sixteenth = np.round(np.array(range(0,4))*64)

    filename = 'Original.zip'


    archive0 = ZipFile('model/Original.zip', 'r')
    archive1 = ZipFile('model/Solutions.zip', 'r')

    for i in range(0,780):
        fileX = 'Original/image_actual_%s.png'%(i)
        image_data = archive0.read(fileX)
        fh = io.BytesIO(image_data)
        img = Image.open(fh).convert('L')

        data_x = np.floor(np.array(img.getdata())/255).reshape(256,256)

        fileY = 'Solutions/image_actual_%s.jpg'%(i)
        image_data = archive1.read(fileY)
        fh = io.BytesIO(image_data)
        img = Image.open(fh).convert('L')

        data_y = np.floor(np.array(img.getdata())/255).reshape(256,256)


        for j in sixteenth:
            for k in sixteenth:
                x = data_x[j:(j+64), k:(k+64)].reshape(1,-1)[0]
                y = data_y[j:(j+64), k:(k+64)].reshape(1,-1)[0]
                X_full.append(data_x[j:(j+64), k:(k+64)])
                Y.append(y)
    # print(sys.argv)
    # filename = sys.argv[1]
    # with ZipFile(filename) as archive:
    #     for entry in archive.infolist():
    #         with archive.open(entry) as file:
    #             img = Image.open(file).convert('L')
    #             data_x = np.floor(np.array(img.getdata())/255).reshape(256,256)
    #             for j in sixteenth:
    #                 for k in sixteenth:
    #                     X_full.append(data_x[j:(j+64), k:(k+64)])
    #
    # filename = 'Solutions.zip'
    # with ZipFile(filename) as archive:
    #     for entry in archive.infolist():
    #         with archive.open(entry) as file:
    #             img = Image.open(file).convert('L')
    #             data_y = np.floor(np.array(img.getdata())/255).reshape(256,256)
    #             for j in sixteenth:
    #                 for k in sixteenth:
    #                     Y.append(data_y[j:(j+64), k:(k+64)].reshape(1,-1)[0])

    X_full = np.array(X_full).reshape((len(X_full), 64,64,1))
    Y = np.array(Y)

    # define the keras model
    model = Sequential()

    model.add(layers.Conv2D(32,(5,5),activation='relu', input_shape=(64,64,1))) # 32 (5,5)
    model.add(layers.MaxPooling2D((3, 3))) # Max (3,3)
    model.add(layers.Conv2D(64, (5, 5), activation='relu')) # 64 (5,5)
    model.add(layers.MaxPooling2D((2, 2))) # Max (2,2)
    model.add(layers.Flatten())

    model.add(Dense(6000, activation='relu')) # 6000
    model.add(Dense(len(Y[0]), activation='relu'))

    model.compile(loss=custom_loss_function, optimizer='adam', metrics=['mse','mae'])
    history = model.fit(X_full, Y, epochs=20, verbose = 1)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

    return model
