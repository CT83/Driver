# Fix error with Keras and TensorFlow
import cv2
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, np
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential


def comma_ai(width, height, channels):
    """ Get the model, this is a slight modification of comma.ai model
    """
    # image size should be (45,160) with 3 color channels
    ch, row, col = channels, height, width

    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - 0.5,
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    # model.add(Convolution2D(3, 1, 1, border_mode="same"))
    # model.add(ELU())

    # Conv Layer1 of 16 filters having size(8, 8) with strides (4,4)
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    # Conv Layer1 of 32 filters having size(5, 5) with strides (2,2)
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    # Conv Layer1 of 64 filters having size(5, 5) with strides (2,2)
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(3))

    return model
