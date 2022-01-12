from utils.functions import  *
from utils.ransac import *
import numpy as np
import matplotlib.image as mpimg
import os
import pickle

input_pth = './images/'
output_pth = './test/'

#HoughLinesP is used to find the sideline. Adjust the parameters for better performance.
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
#threshold = 100  # minimum number of votes (intersections in Hough grid cell)
#min_line_length = 50  # minimum number of pixels making up a line
#max_line_gap = 25  # maximum gap in pixels between connectable line segments

threshold = 150  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 200  # minimum number of pixels making up a line
max_line_gap = 25
##############

def run_sideline_detection(pth):
    original_img = mpimg.imread(pth)
    edges_img, bw = image_preprocessing(original_img)

    lines, line_img = run_hough(edges_img, rho, theta, threshold, min_line_length, max_line_gap)
    sideline_image, found = draw_upper_sideline(lines, bw)
    #print(found)
    #cv2.imwrite('test/test_1.jpg', sideline_image)
    # Look for upper sideline if not found search for lower.
    if found:
        try:
            original_img, contours_used = extract_best_ransac_line(sideline_image, original_img, lower=False)
        except:
            print('No ransac lines found for {}'.format(pth))
    else:
        sideline_image, found = find_lower_sidelines(lines, bw)
        if found:
            try:
                original_img, contours_used = extract_best_ransac_line(sideline_image,
                                                        original_img,
                                                        lower=True)
            except:
                print('No ransac lines found for {}'.format(pth))
    try:
        contours_used
    except:
        contours_used = np.array([[0, 0],[0, 0],[0, 0],[0, 0]])
    return original_img, sideline_image, found, contours_used


def save_cut_frame(img, file_name, output_pth):
    output_pth_full = output_pth + '/{}'.format(file_name)
    cv2.imwrite(output_pth_full, img)
    pass



frame_nums = []
clip_name = ''
for file in (os.listdir(input_pth)):
    if '.jpg' in file:
        frame_nums.append(int(file.split("_")[-1][:-4]))
        clip_name = "_".join([ c for c in file.split("_")[:-1]])

previous_frame_found = False
previous_missing = False
previous_contours_used = None

for frame in sorted(frame_nums):
    file = clip_name + '_' + str(frame) + '.jpg'
    full_file_path = os.path.join(input_pth,file)
    img, sideline_image, found, contours_used = run_sideline_detection(full_file_path)
    to_save = contours_used
    # If no line found but previous frame had line, use previous line
    if previous_frame_found == True and found == False:
        print('not found', full_file_path)
        cv2.fillPoly(img, pts=[previous_contours_used], color=(0, 0, 0))
        to_save = previous_contours_used
    # If line found but significantly different to previous line, use previous line
    if found == True and not previous_contours_used is None and np.allclose(
    contours_used, previous_contours_used, atol=35) == False and previous_frame_found == True:
        print('issue', full_file_path)
        img = mpimg.imread(full_file_path)
        contours_used = previous_contours_used
        cv2.fillPoly(img, pts=[contours_used], color=(0, 0, 0))
    save_cut_frame(img, file, output_pth)
    previous_frame_found = found
    previous_contours_used = contours_used

