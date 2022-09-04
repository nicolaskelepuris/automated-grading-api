import numpy as np
import cv2

## constants ##
width = 800
height = 600

answers = [0, 1, 2, 1, 3, 4, 0, 2, 4, 3]
choices = 5
questions = len(answers)

###

def proccess(exams = ["./images/modif.jpg"]):
    if isinstance(exams[0], str):
        original_img = cv2.imread(exams[0])
    else:
        original_img = cv2.imdecode(exams[0], cv2.IMREAD_UNCHANGED)
    
    gray_img, answers_frame_corner_points = find_answers_frame_corner_points(original_img)

    black_and_white_img = tranform_to_binary_black_and_white_img(gray_img, answers_frame_corner_points)

    black_and_white_answers = split_answer_options(black_and_white_img)

    processed_answers = process_answers(black_and_white_answers)
    print("USER ANSWERS", processed_answers)

    grade = calculate_grade(processed_answers)
    #print("GRADING",grading)

    score = (sum(grade) / questions) * 100
    print(score)

    cv2.waitKey(0)
    return

def calculate_grade(processed_answers):
    grading=[]
    for x in range(0,questions):
        if answers[x] == processed_answers[x]:
            grading.append(1)
        else:
            grading.append(0)
    return grading

def process_answers(answer_options):
    row = 0
    column = 0

    answer_options_non_zero_pixels_count = np.zeros((questions,choices)) # TO STORE THE NON ZERO VALUES OF EACH BOX
    for image in answer_options:
        totalPixels = cv2.countNonZero(image)
        answer_options_non_zero_pixels_count[row][column]= totalPixels
        column += 1
        if (column == choices):
            column = 0; row += 1
    
    #cv2.imshow('boxes', boxes[4])

    processed_answers=[]
    for x in range (0,questions):
        arr = answer_options_non_zero_pixels_count[x]
        myIndexVal = np.where(arr == np.amax(arr))
        processed_answers.append(myIndexVal[0][0])
    return processed_answers

def tranform_to_binary_black_and_white_img(gray_img, answers_frame_corner_points):
    corners = np.float32(answers_frame_corner_points)
    original_corners = np.float32([[0, 0],[width, 0], [0, height],[width, height]])

    matrix = cv2.getPerspectiveTransform(corners, original_corners)
    warp_img = cv2.warpPerspective(gray_img, matrix, (width, height))
    
    threshold_img = cv2.threshold(warp_img, 170, 255, cv2.THRESH_BINARY_INV )[1]
    #cv2.imshow('threshold img', threshold_img)

    return threshold_img

def find_answers_frame_corner_points(original_img):
    resized_img = cv2.resize(original_img, (width, height))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (7,7),1)
    canny_img = cv2.Canny(blur_img, 50, 50)

    contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    answers_frame_contours = get_biggest_rectangle(contours)

    answers_frame_corner_points = reorder(get_corner_points(answers_frame_contours))

    return gray_img, answers_frame_corner_points

def get_corner_points(cont):
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

def split_answer_options(img):
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