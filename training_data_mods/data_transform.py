from random import shuffle

import cv2
import numpy as np
from cv2 import cv2

from collect_data import VERTICES

WIDTH = 50
HEIGHT = 50
START_SAVE_DATA = 2
DATA_RANGE = 1050
PROCESS_BATCH_SIZE = 10

# TRAINING_DATA_NPY_PATH = 'D:\Driver/Training Data/Driver/400_400 Approx Images/training_data-{}.npy'
TRAINING_DATA_NPY_PATH = \
    'D:\MegaSync/Languages/Python/CT83-PC/Driver-Server/computer/training_data/training_data-{}.npy'
# PROCESSED_DATA_NPY_PATH = 'D:\Driver/Training Data/Driver/50_50 Long/balanced/training_data-{}.npy'
PROCESSED_DATA_NPY_PATH = \
    'D:\MegaSync/Languages/Python/CT83-PC/Driver-Server/computer/training_data/processed/training_data-{}.npy'


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
            # input("Press Enter to continue...")

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
    name_ctr = START_SAVE_DATA
    for j in range(1, DATA_RANGE, PROCESS_BATCH_SIZE):
        training_data = []
        try:
            data = combine_all_data(start_data_range=j, data_range=j + PROCESS_BATCH_SIZE)

            print("Data Shape:", len(data))
            for frame_input in data:
                image = frame_input[0]
                keys = frame_input[1]
                image = process_img(image, width=WIDTH, height=HEIGHT)

                preview_image(image)
                print(keys)

                training_data.append([image, keys])
            training_data = balance_data(training_data)
            shuffle(training_data)
            print("Saving ", PROCESSED_DATA_NPY_PATH.format(name_ctr))
            # np.save(PROCESSED_DATA_NPY_PATH.format(name_ctr), training_data)
            name_ctr += 1
        except Exception as e:
            print(e)


def preview_image(image, name="window"):
    cv2.imshow(name, cv2.resize(image, (400, 400)))
    if cv2.waitKey(3) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


def process_camera_image(original_img):
    process_image = original_img
    process_image = cv2.flip(process_image, 0)  # Vertical Flip
    # process_image = imutils.rotate(process_image, 90)
    process_image = cv2.cvtColor(process_image, cv2.COLOR_BGR2RGB)
    return process_image


def process_img(original_img, width=640, height=640):
    processed_img = process_camera_image(original_img)
    # processed_img = cv2.Canny(processed_img, threshold1=25, threshold2=20)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)

    # processed_img = cv2.resize(processed_img, (width, height))
    processed_img = region_of_interest(processed_img, [VERTICES])

    return processed_img


def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


if __name__ == '__main__':
    data_transform()
