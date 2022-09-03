from ast import Return
import numpy as np
import cv2


def proccess(path_gabarito = "./images/gabarito_NICOLAU.jpg", path_alunos = [ "./images/shapes.png", "./images/prova_teste_nicolau_1.jpg"]):
    img = cv2.imread(path_alunos[0]) #1
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #2
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)

    imgFinal = img.copy()

    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(imgFinal, contours, -1, (255, 255, 0), 10)

    biggestRect = get_biggest_rectangle(contours)
    biggestPoints= getCornerPoints(biggestRect)

    cv2.drawContours(imgFinal, biggestPoints, -1, (0, 255, 0), 20)
    
    cv2.imshow('asd', imgFinal)

    cv2.waitKey(0)
    return

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def get_biggest_rectangle(contours):
    max_value = [0, 0] #area, contorno
    for i in contours:
        if cv2.contourArea(i) > 50:
            perimeter = cv2.arcLength(i, True)
            corners = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            if len(corners) == 4:
                area = cv2.contourArea(i)
                if area > max_value[0]:
                    max_value = [area, i]
                    
    return max_value[1]

proccess()