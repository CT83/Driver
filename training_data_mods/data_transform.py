from random import shuffle

import cv2
import numpy as np
from cv2 import cv2

from collect_data import VERTICES

WIDTH = 50
HEIGHT = 50
START_DATA = 6
DATA_RANGE = 149
PROCESS_BATCH_SIZE = 10

TRAINING_DATA_NPY_PATH = 'D:\Driver/Training Data/Driver/400_400 Approx Images/training_data-{}.npy'
PROCESSED_DATA_NPY_PATH = 'D:\Driver/Training Data/Driver/50_50 Long/training_data-{}.npy'


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


def combine_all_data(start_data_range=1, data_range=DATA_RANGE):
    train_data = []
    for j in range(start_data_range, data_range):
        try:
            print("Loading training_data-{}.npy", j)
            inf_from_every_file = np.load(TRAINING_DATA_NPY_PATH.format(j))
            train_data.append(inf_from_every_file)
        except Exception as e:
            print(e)
            print('Failed to Load training_data-{}.npy', j)
    train_data = np.concatenate(train_data)
    return train_data


def data_transform():
    name_ctr = START_DATA
    for j in range(1, DATA_RANGE, PROCESS_BATCH_SIZE):
        training_data = []
        try:
            data = combine_all_data(start_data_range=j, data_range=j + PROCESS_BATCH_SIZE)

            print("Data Shape:", len(data))
            for frame_input in data:
                image = frame_input[0]
                keys = frame_input[1]
                image = process_img(image, width=WIDTH, height=HEIGHT)

                # Augmentation
                # if keys == [1, 0, 0] or keys == [0, 0, 1]:
                #     image_2 = cv2.flip(image, 1)
                #     if keys == [1, 0, 0]:
                #         keys_2 = [0, 0, 1]
                #     else:
                #         keys_2 = [1, 0, 0]
                #     training_data.append([image_2, keys_2])
                #
                # training_data.append([image, keys])

                # preview_image(image)
                training_data.append([image, keys])
            training_data = balance_data(training_data)
            shuffle(training_data)
            print("Saving ", PROCESSED_DATA_NPY_PATH.format(name_ctr))
            np.save(PROCESSED_DATA_NPY_PATH.format(name_ctr), training_data)
            name_ctr += 1
        except Exception as e:
            print(e)


def preview_image(image):
    cv2.imshow('window', image)
    if cv2.waitKey(3) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    data_transform()


def process_img(original_img, width=100, height=100):
    processed_img = original_img
    processed_img = cv2.Canny(processed_img, threshold1=100, threshold2=300)
    # processed_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    processed_img = roi(processed_img, [VERTICES])
    # lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, np.array([]), minLineLength=50, maxLineGap=600000)
    # draw_lines(processed_img, lines)

    # processed_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    # processed_img = cv2.GaussianBlur(processed_img, (5, 5), 1)
    processed_img = cv2.resize(processed_img, (width, height))
    # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    return processed_img


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked