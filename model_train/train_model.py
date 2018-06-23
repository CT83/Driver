from random import shuffle

import numpy as np

from models.alexnet import alexnet
from training_data_mods.data_transform import PROCESSED_DATA_NPY_PATH

TRAINING_DATA_NPY_PATH = 'D:\Training Data\Driver/400_400 Approx Images/balanced_shuffled_resized_100_100/training_data-{}.npy'

WIDTH = 224
HEIGHT = 224
LR = 1e-3
EPOCHS = 18
MODEL_NAME = 'June_21_Only_Grayscale_Same_ROI'

PREV_MODEL = 'June_21_Only_Grayscale_Same_ROI12'
LOAD_MODEL = True

DATA_RANGE = 15
VAL_SIZE = 1000


def combine_all_data(data_range=DATA_RANGE, data_path=PROCESSED_DATA_NPY_PATH):
    train_data = []
    for j in range(1, data_range):
        try:
            print("Loading training_data-{}.npy", j)
            inf_from_every_file = np.load(data_path.format(j))
            train_data.append(inf_from_every_file)
        except Exception as e:
            print(e)
            print('Failed to Load training_data-{}.npy', j)
    train_data = np.concatenate(train_data)
    return train_data


def main():
    model = alexnet(WIDTH, HEIGHT, LR)
    if LOAD_MODEL:
        model.load(PREV_MODEL)
        print('We have loaded a previous model!!!!')

    for epoch in range(EPOCHS):
        # for j in range(1, DATA_RANGE):
        try:
            train_data = combine_all_data(data_range=16)
            print("Train Data Shape:", train_data.shape)
            shuffle(train_data)
            print("Training Data ", str(len(train_data)))
            # train_data = balance_data(train_data)
            train = train_data[:-VAL_SIZE]
            test = train_data[-VAL_SIZE:]
            print("Split | Train :", len(train), " | Test :", len(test))

            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
            test_y = [i[1] for i in test]

            model.fit({'input': X}, {'targets': Y}, n_epoch=1,
                      validation_set=({'input': test_x}, {'targets': test_y}),
                      snapshot_step=2500, show_metric=True, run_id=MODEL_NAME,
                      shuffle=True)

            print("Model Saved", epoch)
            m_name = MODEL_NAME + str(epoch)
            model.save(m_name)
        except NameError as e:
            print(e)


if __name__ == '__main__':
    main()
