import time

import cv2
import numpy as np

from Driver.create_training_data import keys_to_output
from Driver.getkeys import key_check
from Driver.grabscreen import grab_screen

BOX = (10, 25, 646, 509)

# VERTICES = np.array([[0, 430], [50, 300], [470, 400], [620, 430], [600, 480], [0, 480]])
VERTICES = np.array([[0, 300], [620, 300], [640, 400], [0, 400]])
keys_to_record = {'W', 'A', 'D'}


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


def is_controls(keys):
    if set(keys_to_record) & set(keys):
        return True
    else:
        return False


def wait_for(sleep_for, delay):
    for i in list(range(sleep_for))[::-1]:
        print(i + 1)
        time.sleep(delay)


def get_file_size(file_name):
    import os.path
    try:
        return str(os.path.getsize(file_name) >> 20)
    except os.error as e:
        print(e)
        return ""


def main():
    print('Starting...')
    wait_for(4, delay=2)
    file_name = 'F:/' + time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))
    training_data = []
    paused = False
    while True:
        screen = grab_screen(region=BOX)
        screen = process_img(screen)
        cv2.imshow('Car Vision', screen)
        keys = key_check()
        output = keys_to_output(keys)
        if is_controls(keys):
            training_data.append([screen, output])

        if len(training_data) % 10000 == 0 and len(training_data) != 0 or 'P' in keys:
            display_stats(training_data)
            file_name = file_name + "_" + (str(len(training_data)))
            np.save(file_name + '.npy', training_data)
            file_size = get_file_size(file_name + '.npy')
            print(str(len(training_data))
                  + "- Saving to " + file_name + '.npy' +
                  ' File Size:' + file_size + " MB"
                  )

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main()
