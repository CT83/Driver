# create_training_data.py

import os
import time

import cv2
import numpy as np
from cv2 import cv2

from Driver.getkeys import key_check
from Driver.grabscreen import grab_screen

BOX = (10, 25, 646, 509)
VERTICES = np.array([[0, 300], [620, 300], [640, 400], [0, 400]])


def keys_to_output(keys):
    """
    Convert keys to a ...multi-hot... array
    [A,W,D] boolean values.
    """
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


def process_img(original_img):
    processed_img = cv2.Canny(original_img, threshold1=100, threshold2=300)
    processed_img = roi(processed_img, [VERTICES])
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 1)
    return processed_img


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def main():
    file_name = 'training_data1.npy'

    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            screen = grab_screen(region=BOX)
            #img = process_img(screen)
            cv2.imshow('Car Vision', screen)
            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, keys])

            if len(training_data) % 200 == 0:
                print(len(training_data))
                np.save(file_name, training_data)

        keys = key_check()
        if 't' in keys:
            if paused:
                paused = False
                print('Un-Paused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


