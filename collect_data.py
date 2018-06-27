# create_training_data.py

import time
from threading import Thread

import cv2
import numpy as np
from cv2 import cv2

from sentdex.getkeys import key_check
from training_data_mods.data_transform import process_img

BOX = (10, 25, 645, 510)
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
    file_name = 'training_data-1.npy'
    starting_value = 1
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
            if len(training_data) % 250 == 0:
                display_stats(training_data)
                print(len(training_data))

                Thread(target=save_data,
                       kwargs={'file_name': file_name,
                               'training_data': training_data}).start()

                print('Saved', file_name)
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
    np.save('F:\Training Data/' + file_name, training_data)


def wait_countdown():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)


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
    print('Number of Frames', str(len(training_data)))
    print('Forwards : ' + str(len(forwards)))
    print('Lefts    :' + str(len(lefts)))
    print('Rights   :' + str(len(rights)))


if __name__ == "__main__":
    main()
