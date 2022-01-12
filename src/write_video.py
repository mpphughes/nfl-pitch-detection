import os
import cv2

frame_nums = []
clip_name = ''
input_pth = './test/'
for file in (os.listdir(input_pth)):
    if '.jpg' in file:
        frame_nums.append(int(file.split("_")[-1][:-4]))
        clip_name = "_".join([ c for c in file.split("_")[:-1]])

frames = []
for frame in sorted(frame_nums):
    file = clip_name + '_' + str(frame) + '.jpg'
    full_file_path = os.path.join(input_pth,file)
    frames.append(cv2.imread(full_file_path))

size =(frames[0].shape[1], frames[0].shape[0])

out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30,
                      size)
for i in range(len(frames)):
    out.write(frames[i])
out.release()