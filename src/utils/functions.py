import cv2
import numpy as np
import os
import math

def angle_of_line(x1, y1, x2, y2):
    """
    Get angle of line
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    return math.degrees(math.atan2(y2-y1, x2-x1))

def find_lower_boxes(img, file_name, box_dir):
    """
    Function to find players identified at very bottom of screen from box
    information (not in use currently)
    :param img:
    :param file_name:
    :param box_dir:
    :return:
    """
    img_name = file_name.split('_')[-1]
    img_name = '_' + img_name[:-3]
    boxes_lower = False
    lower_boxes = []
    for file in os.listdir(box_dir):
        if img_name in file:
            f = open(box_dir+'{}'.format(file, 'rb'))
            boxes = f.readlines()
            b = boxes[0].split()
            b = b[0].split(',')
            b = [int(b) for b in b]
            if b[3] > img.shape[0]- 200:
                if b[3] - b[1] < 300:#img.shape[1]-20:
                    lower_boxes.append(b)
             #   cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), (0,0,0), -1)

    return lower_boxes

def draw_lower_boxes(img, lower_boxes):
    for b in lower_boxes:
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 0), -1)
    return img

def segment_lines(lines, delta):
    """
    Segment lines based on an angle
    :param lines: array of lines
    :param delta: segmentation angle
    :return: segmented lines in two lists
    """
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2-x1) < delta: # x-values are near; line is vertical
                v_lines.append(line)
            elif abs(y2-y1) < delta: # y-values are near; line is horizontal
                h_lines.append(line)
    return h_lines, v_lines

def is_arr_in_list(myarr, list_arrays):
    """
    Test if an array appears in a list of array
    :param myarr:
    :param list_arrays:
    :return:
    """
    return next((True for elem in list_arrays if elem is myarr), False)


def image_preprocessing(img, save_pth_bw_images=None):
    """
    Preprocess frames for canny edge detection.
    :param img: image
    :param save_pth_bw_images: save location for black white image
    :return: processed image
    """
    # Extract field greens from HV image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # light_green =(50, 75, 100)
    light_green = (25, 50, 75)
    dark_green = (100, 200, 200)

    mask = cv2.inRange(hsv_img, light_green, dark_green)
    result = cv2.bitwise_and(img, img, mask=mask)

    # First, get the gray image and process GaussianBlur.
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Second, process edge detection use Canny.
    low_threshold = 200
    high_threshold = 300
    edges1 = cv2.Canny(blur_gray, low_threshold, high_threshold)

    return edges1, result


def run_hough(edges_img, rho, theta, threshold, min_line_length, max_line_gap):
    """
    Run HouhgLinesP on edge image to find field sidelines
    :param edges_img:
    :param rho:
    :param theta:
    :param threshold:
    :param min_line_length:
    :param max_line_gap:
    :return: lines
    """
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = np.copy(edges_img) * 0
    lines = cv2.HoughLinesP(edges_img, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return lines, line_image

def draw_sideline(lines, img):
    """
    Draws top (upper most sideline)
    :param lines: HoughLinesP result
    :param img: A line image of sideline
    :return:
    """
    h_l, v_l = segment_lines(lines, 90)
    line_image = np.copy(img) * 0
    line_image = line_image + 255
    found = False

    h_l_lst = []
    for line in h_l:
        h_l_lst.append([v for v in line[0]])

    line_lst = []
    for line in lines:
        line_lst.append([v for v in line[0]])

    v_values = []

    # Find where majority of lines are present in image. Aim is to identify
    # where yard lines are on image then we know sideline cannot be in the
    # same space as a yard line!
    for line2 in line_lst:
        if line2 not in h_l_lst:
            start = min(line2[1], line2[3])
            end = max(line2[1], line2[3])
            v_values.append([i for i in range(start, end)])

    v_values_all = [v for sublst in v_values for v in sublst]
    upper_boundary = np.percentile(np.array(v_values_all), 90)
    lower_boundary = np.percentile(np.array(v_values_all), 10)
    found = False
    for line in h_l_lst:
        if line[1] < 600 and line[3] < 600:
            on_field = False
            lower = min(line[1], line[3])
            upper = max(line[1], line[3])
            if lower > lower_boundary:
                on_field = True
            # Check if a yard line is present above or below suspected sideline.
            # If present then found line is on field and not a sideline.
            for line2 in line_lst:
                if line2 not in h_l_lst:
                    if (line2[1] < line[1] * 0.86 and line2[3] > line[
                        3] * 1.1) or (
                            line2[3] < line[1] * 0.86 and line2[1] > line[
                        3] * 1.1):
                        on_field = True
                    # If yard line is right at edge of image then no sideline
                    # must be visible
                    elif (line2[1] < 10 and line2[3] > 100) or (
                            line2[3] < 10 and line2[1] > 100):
                        on_field = True
            # Draw line if suspected sideline not in same space as yardline
            if not on_field:
                found = True
                cv2.line(line_image, (int(line[0]), int(line[1])),
                         (int(line[2]), int(line[3])), (0, 0, 0), 5)

    return line_image, found


def line_preparation(lines):
    """
    helper function
    :param lines:
    :return:
    """
    h_l, v_l = segment_lines(lines, 80)

    h_l_lst = []
    for line in h_l:
        h_l_lst.append([v for v in line[0]])

    line_lst = []
    for line in lines:
        line_lst.append([v for v in line[0]])

    v_values = []
    for line2 in line_lst:
        if line2 not in h_l_lst:
            start = min(line2[1], line2[3])
            end = max(line2[1], line2[3])
            v_values.append([i for i in range(start, end)])

    v_values_all = [v for sublst in v_values for v in sublst]
    upper_boundary = np.percentile(np.array(v_values_all), 90)
    lower_boundary = np.percentile(np.array(v_values_all), 10)

    return h_l_lst, line_lst, lower_boundary, upper_boundary


def draw_lower_sideline(line_image, line, line_lst, h_l_lst, boundary):
    """
    Find and draw lower sideline using same but inverted principle as for upper
    sideline
    :param line_image:
    :param line:
    :param line_lst:
    :param h_l_lst:
    :param boundary:
    :return:
    """
    on_field = False
    found = False
    lower = min(line[1], line[3])
    upper = max(line[1], line[3])

    if lower > boundary:
        on_field = True

    for line2 in line_lst:
        if line2 not in h_l_lst:
            for line2 in line_lst:
                if line2[1] > line[1] * 1.2 or line2[1] > line[3] * 1.2 or \
                        line2[
                            3] > line[1] * 1.2 or line2[3] > line[3] * 1.2:
                    if line2[1] < line[1] * 0.9 or line2[1] < line[3] * 0.9 or \
                            line2[
                                3] < line[1] * 0.9 or line2[3] < line[3] * 0.9:
                        on_field = True
    if not on_field:
        angle = angle_of_line(line[0], line[1], line[2], line[3])
        if abs(angle) < 45:
            found = True
            cv2.line(line_image, (int(line[0]), int(line[1])),
                     (int(line[2]), int(line[3])), (0, 0, 0), 5)
    return line_image, found


def find_lower_sidelines(lines, bw):
    h_l_lst, line_lst, lower_boundary, upper_boundary = line_preparation(lines)
    line_image = np.copy(bw) * 0
    line_image = line_image + 255
    found = False
    # Search for bottom sideline
    for line in h_l_lst:
        if line[1] > 600 and line[3] > 600:
            # Ensure line is not from watermark (score board).
            if line[1] < 950 or line[3] < 950:
                line_image, found = draw_lower_sideline(
                    line_image, line, line_lst, h_l_lst,
                    boundary=upper_boundary)#, upper=True)

    return line_image, found

