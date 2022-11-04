import numpy as np
import cv2

## constants ##
width = 850
height = 1210
id_digits_options = 10
###

def process(exams = ["./images/p3.png"], correct_answers = [3, 1, 0, 1, 2, 3, 4, 3, 4, 0], choices_per_question_count = 5, id_digits_count = 9, question_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
    id_digits_count += 1 # adiciona 1 row que será ignorada (header Matrícula)
    questions_count = len(correct_answers)
    print("correct_answers", correct_answers)

    processed_exams = process_exams(exams, choices_per_question_count, questions_count, id_digits_count)
    answers = list(map(lambda p: p[0], processed_exams))
    ids = list(map(lambda p: p[1][1:], processed_exams))
    print("exams_answers:", answers)
    print("exams_ids:", ids)

    compared_answers = compare_exams_to_answers(answers, correct_answers)
    print("compared_answers:", compared_answers)

    compared_answers_and_ids = []
    for i, x in enumerate(compared_answers):
        compared_answers_and_ids.append({ "compared_answers": x, "id": ids[i] })

    return { "data": list(map(lambda exam: format_answers(exam, question_weights), compared_answers_and_ids)) }

def format_answers(processed_exam, question_weights):
    return {
        "compared_answers": processed_exam["compared_answers"],
        "score": (sum(question_weights[i] for i in range(0, len(processed_exam["compared_answers"])) if processed_exam["compared_answers"][i]) / sum(question_weights)) * 10,
        "correct_count": sum(1 for i in range(0, len(processed_exam["compared_answers"])) if processed_exam["compared_answers"][i]),
        "id": processed_exam["id"]
    }

def process_exams(exams, choices_count, questions_count, id_digits_count):
    result = []
    for exam in exams:
        if isinstance(exam, str):
            original_img = cv2.imread(exam)
        else:
            original_img = cv2.imdecode(exam, cv2.IMREAD_UNCHANGED)
        
        answers, id = process_image(choices_count, questions_count, original_img, id_digits_count)

        result.append([answers, id])

    return result

def process_image(choices_count, questions_count, original_img, id_digits_count):
    gray_img, answers_frame, id_frame = find_answers_and_id_frame_corner_points(original_img)

    processed_answers = frame_to_marked_options(gray_img, answers_frame, questions_count, choices_count, ignore_first_column = True)

    processed_id = frame_to_marked_options(gray_img, id_frame, id_digits_count, id_digits_options)

    return processed_answers, processed_id

def frame_to_marked_options(gray_img, frame, rows_count, cols_count, ignore_first_column = False):
    black_and_white_answers_frame_img = tranform_to_binary_black_and_white_img(gray_img, frame)

    (h, w) = black_and_white_answers_frame_img.shape
    black_and_white_answers_frame_img =  black_and_white_answers_frame_img[7:(h - 23), 7:(w - 5)]    

    black_and_white_answers = split_rows_and_columns(black_and_white_answers_frame_img, rows_count, cols_count, ignore_first_column)

    return process_answers(black_and_white_answers, rows_count, cols_count)

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

def find_answers_and_id_frame_corner_points(original_img):
    resized_img = cv2.resize(original_img, (width, height))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (7,7),1)
    canny_img = cv2.Canny(blur_img, 50, 50)

    contours, _ = cv2.findContours(canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(resized_img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("original_img,", resized_img)
    # cv2.waitKey()
    frames = get_biggest_rectangles(contours)
    # cv2.drawContours(resized_img, frames, -1, (0, 255, 0), 10)
    # cv2.imshow("original_img,", resized_img)
    # cv2.waitKey()

    frames = list(map(lambda frame: reorder(get_corner_points(frame)), frames))
    # cv2.drawContours(resized_img, frames, -1, (0, 255, 0), 10)
    # cv2.imshow("original_img,", resized_img)
    # cv2.waitKey()

    answers_frame, id_frame = separate_frames(frames)
    # cv2.drawContours(resized_img, [answers_frame, id_frame], -1, (0, 255, 0), 10)
    # cv2.imshow("original_img,", resized_img)
    # cv2.waitKey()

    return gray_img, answers_frame, id_frame

def separate_frames(frames):
    if frames[0][0][0][1] < frames[1][0][0][1]:
        id_frame = frames[0]
        answers_frame = frames[1]
    else:
        id_frame = frames[1]
        answers_frame = frames[0]
    return answers_frame, id_frame

def get_corner_points(contours):
    perimeter = cv2.arcLength(contours, True)
    corner_points = cv2.approxPolyDP(contours, 0.02 * perimeter, True)

    return corner_points

def get_biggest_rectangles(contours):
    rectangles = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if not has_min_area(area): continue
        if not is_rectangle(contour): continue

        rectangles.append([area, contour])

    rectangles.sort(key = lambda k: k[0], reverse = True)
    return list(map(lambda rectangle: rectangle[1], rectangles[:2]))

def to_area_and_contour(contour):
    [cv2.contourArea(contour), contour]

def has_min_area(area):
    return area > 50

def is_rectangle(contour):
    perimeter = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return len(corners) == 4

def split_rows_and_columns(img, questions_count, choices_count, ignore_first_column = False):
    if ignore_first_column:
        choices_count = choices_count + 1
    
    img = cv2.resize(img, (round(width / choices_count) * choices_count, round(height / questions_count) * questions_count))

    rows = np.vsplit(img, questions_count)
    options = []
    for r in rows:
        cols = np.hsplit(r, choices_count)
        if ignore_first_column:
            cols = cols[1:]
        for choice_option in cols:
            options.append(choice_option)

    return options

def reorder(contours):
    contours = contours.reshape((4, 2)) # Format to an array of 4 contours (each one is an array of 2 values)

    reordered = np.zeros((4, 1, 2), np.int32) # Initialize result with zeros

    sum = contours.sum(1)
    reordered[0] = contours[np.argmin(sum)] # top left
    reordered[3] = contours[np.argmax(sum)] # bottom right

    diff = np.diff(contours, axis=1)
    reordered[1] = contours[np.argmin(diff)] # top right
    reordered[2] = contours[np.argmax(diff)] # bottom left

    return reordered

# print(process())