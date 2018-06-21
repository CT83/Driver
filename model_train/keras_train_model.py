import json
from random import shuffle

# Fix error with Keras and TensorFlow
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from model_train.train_model import combine_all_data

WIDTH = 100
HEIGHT = 100
LR = 1e-3
EPOCHS = 40
MODEL_NAME = 'AlexNet_Keras_21th_June'

PREV_MODEL = 'model_alexnet-702'

LOAD_MODEL = False
hm_data = 5
VAL_SIZE = 1000


def batch_data_generator(x, y, batch_size):
    while True:
        yield x, y


def main():
    train_data = None
    tensorboard = TensorBoard(log_dir="log/" + MODEL_NAME)
    try:
        from training_data_mods.data_transform import PROCESSED_DATA_NPY_PATH
        train_data = combine_all_data(data_range=15,
                                      data_path=PROCESSED_DATA_NPY_PATH)
        shuffle(train_data)
    except NameError as e:
        print(e)
    print("Train Data Shape:", train_data.shape)
    X = np.array([i[0] for i in train_data]).reshape(-1, WIDTH, HEIGHT, 1)
    # X = np.array([i[0] for i in train_data])
    Y = [i[1] for i in train_data]
    x_train, x_val, y_train, y_val = train_test_split(X, Y,
                                                      test_size=0.15,
                                                      random_state=1111)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    print("Train data : (X, Y) ", x_train.shape, y_train.shape)
    print("Validation data : (X, Y) ", x_val.shape, y_val.shape)

    nb_epoch = 1
    batch_size = 16

    # Create the Model
    from models.ImageModels.AlexNet import create_model
    xy, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    from keras import Model
    model = Model(input=img_input,
                  output=[xy])

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print('Model Compiled')

    # if LOAD_MODEL:
    #     print('We have loaded a previous model.')
    #     model.load_weights(MODEL_NAME + '.h5')

    model.summary()
    checkpointer = ModelCheckpoint(filepath=MODEL_NAME + ".h5", verbose=1, save_best_only=True)

    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=10, verbose=1,
                        shuffle=True,
                        callbacks=[tensorboard],
                        validation_data=(x_val, y_val))
    # Save weights
    model.save_weights(MODEL_NAME + ".h5", True)
    # Save model architecture
    with open('./model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)


if __name__ == '__main__':
    main()
