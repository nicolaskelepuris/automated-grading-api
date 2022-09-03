import numpy as np
import cv2

## constants ##
defined_width = 800
defined_height = 600

answers = [0, 1, 2, 1, 3, 4, 0, 2, 4, 3]
choices = 5
questions = len(answers)

###

def proccess(path_gabarito = "./images/gabarito_NICOLAU.jpg", path_alunos = [  "./images/modif.jpg" ]):
    fullimg = cv2.imread(path_alunos[0]) #1
    img = cv2.resize(fullimg, (defined_width, defined_height))

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #2
    imgBlur = cv2.GaussianBlur(imgGray, (7,7),1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)

    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggestRect = get_biggest_rectangle(contours)
    biggestPoints= getCornerPoints(biggestRect)
    imgFinal = img.copy()

    #cv2.drawContours(imgFinal, biggestPoints, -1, (0, 255, 0), 20)
    #cv2.drawContours(imgFinal, contours, -1, (0, 255, 0), 20)
    #cv2.imshow('imgfinal', imgFinal)

    biggestPoints = reorder(biggestPoints) # REORDER FOR WARPING
    cv2.drawContours(imgFinal, biggestPoints, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
    pts1 = np.float32(biggestPoints) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[defined_width, 0], [0, defined_height],[defined_width, defined_height]]) # PREPARE POINTS FOR WARP

    matrix = cv2.getPerspectiveTransform(pts1, pts2) # GET TRANSFORMATION MATRIX
    imgWarpColored = cv2.warpPerspective(imgGray, matrix, (defined_width, defined_height))
    
    imgThresh = cv2.threshold(imgWarpColored, 170, 255,cv2.THRESH_BINARY_INV )[1]
    cv2.imshow('imgThresh', imgThresh)

    boxes = splitBoxes(imgThresh) # GET INDIVIDUAL BOXES

    countR=0
    countC=0

    myPixelVal = np.zeros((questions,choices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC]= totalPixels
        countC += 1
        if (countC==choices):countC=0;countR +=1
    
    #cv2.imshow('boxes', boxes[4])

    myIndex=[]
    for x in range (0,questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        myIndex.append(myIndexVal[0][0])
    print("USER ANSWERS",myIndex)
    #print("GABARITO    ", answers)

    grading=[]
    for x in range(0,questions):
        if answers[x] == myIndex[x]:
            grading.append(1)
        else:grading.append(0)
    #print("GRADING",grading)

    score = (sum(grading)/questions)*100 # FINAL GRADE
    print(score)

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

def splitBoxes(img):
    rows = np.vsplit(img, len(answers))
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,choices)
        for box in cols:
            boxes.append(box)
    return boxes

def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    #print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    #print(add)
    #print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew

proccess()