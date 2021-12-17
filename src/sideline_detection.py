from utils.functions import  *
from utils.ransac import *
import numpy as np
import matplotlib.image as mpimg
import os

input_pth = './images/'
output_pth = './test/'

#HoughLinesP is used to find the sideline. Adjust the parameters for better performance.
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 100  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 25  # maximum gap in pixels between connectable line segments

##############

def run_sideline_detection(pth):
    original_img = mpimg.imread(pth)
    edges_img, bw = image_preprocessing(original_img)

    lines, line_img = run_hough(edges_img, rho, theta, threshold, min_line_length, max_line_gap)
    sideline_image, found = draw_sideline(lines, bw)
    # Look for upper sideline if not found search for lower.
    if found:
        try:
            original_img = extract_best_ransac_line(sideline_image, original_img, lower=False)
        except:
            print('No ransac lines found for {}'.format(pth))
    else:
        lines, line_img = run_hough(edges_img, rho, theta, 150,
                                    min_line_length, max_line_gap)
        sideline_image, found = find_lower_sidelines(lines, bw)
        # cv2.imwrite('side.png', sideline_image)
        if found:
            try:
                original_img = extract_best_ransac_line(sideline_image,
                                                        original_img,
                                                        lower=True)
            except:
                print('No ransac lines found for {}'.format(pth))
    return original_img


def save_cut_frame(img, file_name, output_pth):
    output_pth_full = output_pth + '/{}'.format(file_name)
    cv2.imwrite(output_pth_full, img)
    pass



for file in os.listdir(input_pth):
    if '.jpg' in file:
        full_file_path = os.path.join(input_pth,file)
        img = run_sideline_detection(full_file_path)
        save_cut_frame(img, file, output_pth)