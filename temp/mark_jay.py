import time

import cv2
import numpy as np

from collect_data import display_stats, process_img
from temp.create_training_data import keys_to_output
from sentdex.getkeys import key_check
from sentdex.grabscreen import grab_screen

BOX = (10, 25, 646, 509)

# VERTICES = np.array([[0, 430], [50, 300], [470, 400], [620, 430], [600, 480], [0, 480]])
VERTICES = np.array([[0, 300], [620, 300], [640, 400], [0, 400]])
keys_to_record = {'W', 'A', 'D'}


def draw_lines(img, lines):
    try:
        for line in lines:
            coords = line[0]
            cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255, 255, 255], 3)
    except TypeError:
        print("NoneType")


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
    # wait_for(4, delay=2)
    file_name = 'E:/' + time.strftime('%Y_%m_%d %H_%M_%S', time.localtime(time.time()))
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


if __name__ == "__main__":
    main()
