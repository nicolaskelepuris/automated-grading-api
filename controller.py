import numpy as np
import cv2

## constants ##
width = 800
height = 600
###

def proccess(exams = ["./images/modif.jpg"], answers = [0, 1, 2, 1, 3, 4, 0, 2, 4, 3], choices_count = 5):
    questions_count = len(answers)
    if isinstance(exams[0], str):
        original_img = cv2.imread(exams[0])
    else:
        original_img = cv2.imdecode(exams[0], cv2.IMREAD_UNCHANGED)
    
    gray_img, answers_frame_corner_points = find_answers_frame_corner_points(original_img)

    black_and_white_img = tranform_to_binary_black_and_white_img(gray_img, answers_frame_corner_points)

    black_and_white_answers = split_answer_options(black_and_white_img, questions_count, choices_count)

    processed_answers = process_answers(black_and_white_answers, questions_count, choices_count)
    print("USER ANSWERS:", processed_answers)

    correct_answers_count = get_correct_answers_count(processed_answers, answers)
    #print("correct_answers_count:", correct_answers_count)

    score = (correct_answers_count / questions_count) * 100
    print("score:", score)

    cv2.waitKey(0)
    return

def get_correct_answers_count(processed_answers, answers):   
    return sum(1 for i in range(0, len(answers)) if answers[i] == processed_answers[i])

def process_answers(answer_options, questions_count, choices_count):    
    choices_non_zero_pixels_count = list(map(count_non_zero_pixels, answer_options))
    questions = separate_in_questions(choices_non_zero_pixels_count, questions_count, choices_count)

    return list(map(get_answer_index, questions))

def count_non_zero_pixels(image):
    return cv2.countNonZero(image)

def get_answer_index(question_answers):
    return question_answers.index(max(question_answers))

def separate_in_questions(answer_options_non_zero_pixels_count, questions_count, choices_count):
    result = []
    for i in range (0, questions_count):
        start_of_question = choices_count * i
        end_of_question = start_of_question + choices_count
        result.append(answer_options_non_zero_pixels_count[start_of_question : end_of_question])

    return result


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

def get_corner_points(contours):
    perimeter = cv2.arcLength(contours, True)
    corner_points = cv2.approxPolyDP(contours, 0.02 * perimeter, True)

    return corner_points

def get_biggest_rectangle(contours):
    max_value = [0, 0] #area, contorno
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if not has_min_area(area): continue
        if not is_rectangle(contour): continue

        if area > max_value[0]:
            max_value = [area, contour]

    return max_value[1]

def to_area_and_contour(contour):
    [cv2.contourArea(contour), contour]

def has_min_area(area):
    return area > 50

def is_rectangle(contour):
    perimeter = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return len(corners) == 4

def split_answer_options(img, questions_count, choices_count):
    rows = np.vsplit(img, questions_count)
    options = []
    for r in rows:
        cols = np.hsplit(r, choices_count)
        for box in cols:
            options.append(box)

    return options

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