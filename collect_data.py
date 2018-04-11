# create_training_data.py

import os
import time
from threading import Thread

import cv2
import numpy as np
from cv2 import cv2

from sentdex.getkeys import key_check

BOX = (10, 25, 646, 509)
VERTICES = np.array([[0, 300], [620, 300], [640, 400], [0, 400]])


def keys_to_output(keys):
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


def main():
    file_name = 'training_data.npy'
    starting_value = 1

    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        training_data = list(np.load(file_name))
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    wait_countdown()

    paused = False
    while True:

        if not paused:
            from sentdex.grabscreen import grab_screen
            screen = grab_screen(region=BOX)
            img = process_img(screen)
            cv2.imshow('window', cv2.resize(img, (300, 300)))

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([img, output])

            if len(training_data) % 100 == 0:
                display_stats(training_data)
            if len(training_data) % 10000 == 0:
                display_stats(training_data)
                print(len(training_data))

                Thread(target=save_data,
                       kwargs={'file_name': file_name,
                               'training_data': training_data}).start()

                # save_data(file_name, training_data)
                print('SAVED!')
                training_data = []
                starting_value += 1
                file_name = 'training_data-{}.npy'.format(starting_value)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

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


def save_data(file_name, training_data):
    np.save(file_name, training_data)


def wait_countdown():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)


if __name__ == "__main__":
    main()


def display_stats(training_data):
    lefts = []
    rights = []
    forwards = []
    for data in training_data:
        img = data[0]
        choice = data[1]

        if choice == [1, 0, 0]:
            lefts.append([img, choice])
        elif choice == [0, 1, 0]:
            forwards.append([img, choice])
        elif choice == [0, 0, 1]:
            rights.append([img, choice])
    print(str(len(training_data)))
    print('Forwards : ' + str(len(forwards)))
    print('Lefts    :' + str(len(lefts)))
    print('Rights   :' + str(len(rights)))


def process_img(original_img):
    # processed_img = cv2.Canny(original_img, threshold1=100, threshold2=300)
    # processed_img = roi(original_img, [VERTICES])
    # lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 180, np.array([]), minLineLength=50, maxLineGap=600000)
    # draw_lines(processed_img, lines)

    processed_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    # processed_img = cv2.GaussianBlur(processed_img, (5, 5), 1)
    processed_img = cv2.resize(processed_img, (50, 50))

    return processed_img


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked