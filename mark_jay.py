import cv2
from directkeys import PressKey, ReleaseKey
from grabscreen import grab_screen
import numpy as np
import pyautogui
from sklearn.cluster import KMeans
import threading
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D
import random

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
VERTICES = np.array([[0, 300], [620, 300], [640, 500], [0, 500]])

for i in range(3, 0, -1):
    time.sleep(.4)
    print(i)


def t_key(key_a, key_b, key_c):
    PressKey(key_a)
    ReleaseKey(key_b)
    ReleaseKey(key_c)
    time.sleep(keytime)


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_img):
    processed_img = cv2.Canny(original_img,
                              threshold1=100, threshold2=300)
    processed_img = roi(processed_img, [VERTICES])
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    processed_img = cv2.resize(processed_img, (160, 120))
    # cv2.imshow('window2', processed_img)
    try:
        img = processed_img
        lines = cv2.HoughLinesP(img, 1,
                                np.pi / 180, 50, np.array([]), 1, 80)
        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 1)

        cv2.imshow('hough', img)

    except Exception as e:
        print(e)

    return processed_img


def main():
    while True:
        ti = time.time()
        screen = grab_screen(region=BOX)
        # cv2.imshow('window', cv2.cvtColor(screen,
        # cv2.COLOR_BGR2RGB))

        new_screen = process_img(screen)

        print('{:.2f} FPS'.format(1 / (time.time() - ti)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
