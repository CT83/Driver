from random import shuffle

import numpy as np
from cv2 import cv2

from collect_data import save_data

WIDTH = 50
HEIGHT = 50
LR = 1e-3
EPOCHS = 3
hm_data = 5
DATA_RANGE = 152


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


def combine_all_data(data_range=DATA_RANGE):
    train_data = []
    for j in range(1, data_range):
        try:
            print("Loading training_data-{}.npy", j)
            inf_from_every_file = np.load('training_data-{}.npy'.format(j))
            train_data.append(inf_from_every_file)
        except Exception as e:
            print(e)
            print('Failed to Load training_data-{}.npy', j)
    train_data = np.concatenate(train_data)
    return train_data


def data_transform():
    from collect_data import process_img

    for j in range(1, DATA_RANGE):
        training_data = []
        try:
            print("Loading training_data-{}.npy", j)
            data = np.load('F:\Training Data/training_data-{}.npy'.format(j))
            for frame_input in data:
                image = frame_input[0]
                keys = frame_input[1]
                image = process_img(image)
                training_data.append([image, keys])

                preview_image(image)

            save_data('processed/training_data-{}.npy'.format(j),
                      training_data)
        except Exception as e:
            print(e)


def preview_image(image):
    cv2.imshow('window', image)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def main():
    for j in range(1, DATA_RANGE):
        try:
            print("Loading training_data-{}.npy", j)
            data = np.load('F:\Training Data/training_data-{}.npy'.format(j))
            for frame_input in data:
                image = frame_input[0]
                keys = frame_input[1]
                cv2.imshow('window', image)
                print(keys)
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # main()
    data_transform()
