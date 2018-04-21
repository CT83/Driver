from random import shuffle

import numpy as np

from alexnet import alexnet

WIDTH = 50
HEIGHT = 50
LR = 1e-3
EPOCHS = 20
MODEL_NAME = 'April_21_Model_1'

PREV_MODEL = 'model_alexnet-816'

LOAD_MODEL = True

model = alexnet(WIDTH, HEIGHT, LR)
if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')


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

    forwards = forwards[:len(lefts)][:len(rights)]
    lefts = lefts[:len(forwards)]
    rights = rights[:len(forwards)]

    final_data = forwards + lefts + rights
    shuffle(final_data)
    return final_data


hm_data = 5
for i in range(EPOCHS):
    for i in range(1, hm_data + 1, 10):

        train_data = []

        for j in range(i, i + 10):
            try:
                print("Loading 50_50 Long/training_data-{}.npy", j)
                inf_from_every_file = np.load('D:/Training Data/Driver/50_50 Long/training_data-{}.npy'.format(j))
                train_data.append(inf_from_every_file)
            except Exception as e:
                print(e)
                print('Failted to Load training_data-{}.npy', j)

        train_data = np.concatenate(train_data)
        print("Training Data ", str(len(train_data)), " Data ", str(train_data))
        train_data = balance_data(train_data)
        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
        test_y = [i[1] for i in test]

        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}),
                  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

        print("Model Saved", i)
        model.save(MODEL_NAME)

        # tensorboard --logdir=foo:C:/
