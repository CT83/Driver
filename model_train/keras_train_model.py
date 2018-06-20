import json
from random import shuffle

# Fix error with Keras and TensorFlow
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from model_train.train_model import combine_all_data
from models.comma_ai import comma_ai

WIDTH = 100
HEIGHT = 100
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'Comma_Ai_18th_June'

PREV_MODEL = 'model_alexnet-702'

LOAD_MODEL = False
hm_data = 5
VAL_SIZE = 1000


def batch_data_generator(x, y, batch_size):
    while True:
        yield x, y


def main():
    train_data = None
    try:
        from training_data_mods.data_transform import PROCESSED_DATA_NPY_PATH
        train_data = combine_all_data(data_range=7, data_path=PROCESSED_DATA_NPY_PATH)
        shuffle(train_data)
    except NameError as e:
        print(e)
    print("Train Data Shape:", train_data.shape)
    X = np.array([i[0] for i in train_data]).reshape(-1, WIDTH, HEIGHT, 1)
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
    model = comma_ai(WIDTH, HEIGHT, channels=1)
    adam = Adam(lr=1e-12)
    model.compile(optimizer=adam, loss="mse")
    tensorboard = TensorBoard(log_dir="log/" + MODEL_NAME)
    if LOAD_MODEL:
        print('We have loaded a previous model.')
        model.load_weights(MODEL_NAME + '.h5')
    model.summary()
    checkpointer = ModelCheckpoint(filepath=MODEL_NAME + ".h5", verbose=1, save_best_only=True)

    # history = model.fit_generator(
    #     batch_data_generator(x_train, y_train, batch_size),
    #     samples_per_epoch=((len(y_train) // batch_size) * batch_size) * 2,
    #     nb_epoch=nb_epoch,
    #     verbose=1,
    #     validation_data=batch_data_generator(x_val, y_val, batch_size),
    #     nb_val_samples=((len(y_val) // batch_size) * batch_size) * 2,
    #     shuffle=True,
    #     callbacks=[tensorboard],
    # )
    history = model.fit(x_train, y_train, epochs=150, batch_size=10, verbose=1,
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
