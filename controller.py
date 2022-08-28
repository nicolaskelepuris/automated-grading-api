import numpy as np
import cv2


def read(path):
    return cv2.imread(path)

def togray(coloredImg):
    imgGray = cv2.cvtColor(coloredImg, cv2.COLOR_BGR2GRAY)
    return imgGray

def show(filename, img):
    cv2.imshow(f'{filename}', img)

def wait(milisec):
    return cv2.waitKey(milisec)