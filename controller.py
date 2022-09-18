import numpy as np
import cv2

## constants ##
width = 800
height = 600
###

def process(exams = ["./images/modif.jpg"], correct_answers = [0, 1, 2, 1, 3, 4, 0, 2, 4, 3], choices_per_question_count = 5):
    questions_count = len(correct_answers)
    print("correct_answers", correct_answers)

    exams_answers = process_exams(exams, choices_per_question_count, questions_count)
    print("exams_answers:", exams_answers)

    compared_answers = compare_exams_to_answers(exams_answers, correct_answers)
    print("compared_answers:", compared_answers)

    scores = calculate_scores(questions_count, compared_answers)
    print("scores:", scores)
    format_answers(compared_answers)

    # cv2.waitKey(0)
    return { "data": list(map(format_answers, compared_answers)) }

def calculate_scores(questions_count, compared_answers):
    scores = list(map(lambda processed_exam: (sum(1 for i in range(0, len(processed_exam)) if processed_exam[i]) / questions_count) * 100, compared_answers))
    return scores

def format_answers(processed_exam):
    return { "compared_answers": processed_exam, "correct_count": sum(1 for i in range(0, len(processed_exam)) if processed_exam[i]) }

def process_exams(exams, choices_count, questions_count):
    exams_answers = []
    for exam in exams:
        if isinstance(exam, str):
            original_img = cv2.imread(exam)
        else:
            original_img = cv2.imdecode(exam, cv2.IMREAD_UNCHANGED)
        
        result = process_image(choices_count, questions_count, original_img)

        exams_answers.append(result)
    return exams_answers

def process_image(choices_count, questions_count, original_img):
    gray_img, answers_frame_corner_points = find_answers_frame_corner_points(original_img)

    black_and_white_img = tranform_to_binary_black_and_white_img(gray_img, answers_frame_corner_points)

    black_and_white_answers = split_answer_options(black_and_white_img, questions_count, choices_count)

    processed_answers = process_answers(black_and_white_answers, questions_count, choices_count)
    return processed_answers

def compare_exams_to_answers(processed_answers, answers):
    return list(map(lambda a: list(compare_exam(a, answers)), processed_answers))

def compare_exam(exam_answers, answers):
    for i in range(0, len(answers)):
        yield answers[i] == exam_answers[i]

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
    # cv2.drawContours(resized_img, contours, -1, (0, 255, 0), 10)
    # cv2.imshow("original_img,", resized_img)
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
        for choice_option in cols:
            options.append(choice_option)

    return options

def reorder(contours):
    contours = contours.reshape((4, 2)) # Format to an array of 4 contours (each one is an array of 2 values)

    reordered = np.zeros((4, 1, 2), np.int32) # Initialize result with zeros

    sum = contours.sum(1)
    reordered[0] = contours[np.argmin(sum)]  # bottom left
    reordered[3] = contours[np.argmax(sum)]   # top right

    diff = np.diff(contours, axis=1)
    reordered[1] = contours[np.argmin(diff)]  # bottom right
    reordered[2] = contours[np.argmax(diff)] # top left

    return reordered

print(process())