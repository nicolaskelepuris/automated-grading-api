import numpy as np
import cv2


def proccess(path_gabarito = "./images/gabarito_NICOLAU.jpg", path_alunos = [ "./images/prova_teste_nicolau_1.jpg"]):
    coloredImg = read(path_gabarito)
    grayImg = togray(coloredImg)
    show('gab', grayImg)
    wait(1000)
    return

def read(path):
    return cv2.imread(path) #1

def togray(coloredImg):
    imgGray = cv2.cvtColor(coloredImg, cv2.COLOR_BGR2GRAY)
    return imgGray #2

def show(filename, img): 
    cv2.imshow(f'{filename}', img) #3

def wait(milisec):
    return cv2.waitKey(milisec) #4


proccess()