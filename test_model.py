# test_model.py

import time

import numpy as np
from cv2 import cv2

from alexnet import alexnet
from directkeys import ReleaseKey, A, W, D, PressKey
from getkeys import key_check
from grabscreen import grab_screen

WIDTH = 60
HEIGHT = 60
LR = 1e-3
EPOCHS = 10
# MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2', EPOCHS)
MODEL_NAME = "Feb_23_Model"

t_time = 0.09
BOX = (10, 25, 646, 509)
VERTICES = np.array([[0, 300], [620, 300], [640, 400], [0, 400]])


def straight():
    ##    if random.randrange(4) == 2:
    ##        ReleaseKey(W)
    ##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    PressKey(W)
    PressKey(A)
    # ReleaseKey(W)
    ReleaseKey(D)
    # ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)


def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    # ReleaseKey(W)
    # ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)


def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while (True):

        if not paused:
            # 800x600 windowed mode
            # screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            screen = grab_screen(region=BOX)
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()

            from mark_jay import process_img
            screen = process_img(screen)

            cv2.imshow('Car Vision', screen)
            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
            print(prediction)

            turn_thresh = .75
            fwd_thresh = 0.70
            cv2.waitKey(25)
            if prediction[1] > fwd_thresh:
                straight()
                # print("straight")
            elif prediction[0] > turn_thresh:
                left()
                # print("left")
            elif prediction[2] > turn_thresh:
                right()
                # print("right")
            else:
                straight()
                # print("straight")

        keys = key_check()

        # p pauses game and can get annoying.
        if 'P' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)


if __name__ == "__main__":
    main()
