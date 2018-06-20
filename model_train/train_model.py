from random import shuffle

import numpy as np

from models.alexnet import alexnet
from training_data_mods.data_transform import PROCESSED_DATA_NPY_PATH

TRAINING_DATA_NPY_PATH = 'D:\Training Data\Driver/400_400 Approx Images/balanced_shuffled_resized_100_100/training_data-{}.npy'

WIDTH = 100
HEIGHT = 100
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'June_17_Model'

PREV_MODEL = 'model_alexnet-702'

LOAD_MODEL = True
hm_data = 5
DATA_RANGE = 150
VAL_SIZE=1000


def balance_data(train_data):
    lefts = []
    rights = []
    forwards = []

    for data in train_data:
        img = data[0]
        choice = data[1]

        if choice == [1, 0, 0]:
            lefts.append([img, choice])
        elif choice == [0, 1, 0]:
            forwards.append([img, choice])
        elif choice == [0, 0, 1]:
            rights.append([img, choice])
        else:
            print('no matches')
            input("Press Enter to continue...")

    print("Lefts:", len(lefts), " Forwards:", len(forwards), " Rights:", len(rights))
    forwards = forwards[:len(lefts)][:len(rights)]
    lefts = lefts[:len(forwards)]
    rights = rights[:len(forwards)]

    final_data = forwards + lefts + rights
    shuffle(final_data)
    return final_data


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
            train_data = combine_all_data(data_range=7)
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
            model.save(MODEL_NAME + str(epoch))
        except NameError as e:
            print(e)


if __name__ == '__main__':
    main()
