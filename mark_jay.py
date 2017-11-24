import time

import cv2
import numpy as np
import os
from Driver.create_training_data import keys_to_output
from Driver.getkeys import key_check
from Driver.grabscreen import grab_screen

uhd_x = 646
uhd_y = 509
# BOX = (uhd_x, uhd_y, uhd_x + 800, uhd_y + 600)
BOX = (10, 25, 646, 509)

hood_y = 410
horizon_y = 260
side_y = 50
keytime = 0.1

STRAIGHT = 0x11
RIGHT = 0x20
LEFT = 0x1E

# VERTICES = np.array([[0, 430], [50, 300], [470, 400], [620, 430], [600, 480], [0, 480]])
VERTICES = np.array([[0, 300], [620, 300], [640, 400], [0, 400]])
keys_to_record = {'W', 'A', 'S', 'D'}


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_img):
    processed_img = cv2.Canny(original_img, threshold1=100, threshold2=300)
    processed_img = roi(processed_img, [VERTICES])
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 1)
    processed_img = cv2.resize(processed_img, (60, 60))
    return processed_img


def display_stats():
    pass


def is_controls(keys):
    if set(keys_to_record) & set(keys):
        return True
    else:
        return False


def main():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(2)

    # file_name = 'training_data.npy'
    file_name = time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))
    file_name = 'F:/' + file_name
    print('Starting...')
    training_data = []

    while True:
        screen = grab_screen(region=BOX)
        screen = process_img(screen)
        cv2.imshow('Car Vision', screen)
        keys = key_check()
        output = keys_to_output(keys)
        if is_controls(keys):
            training_data.append([screen, output])

        if len(training_data) % 200 == 0 and len(training_data) != 0:
            file_name = file_name + "_" + (str(len(training_data)))
            print("Saving to " + file_name)
            print(len(training_data))
            np.save(file_name, training_data)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


main()
