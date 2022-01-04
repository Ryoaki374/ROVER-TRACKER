# track2.py
"""REFERENCE
https://qiita.com/fugunoko/items/7e5056449e172cbeadd9
https://hk29.hatenablog.jp/entry/2020/02/01/162533
O'REILLY Open CV
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

file_path = 'MOV_1189.mp4'
output_filename = '220103-rover_detect-MOV_1189.mp4'
window_name = 'frame'
delay = 1


def Find_contours(img):
    """
    cvtColorで色空間変えれば2値化した画像でも色乗せれる
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gray scale
    ret, img_thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_OTSU)  # OTSU method

    contours, hierarchy = cv2.findContours(
        img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find ROVER's contours

    contours_ARRAY = np.array(contours)
    x_point = np.mean(contours_ARRAY[0].T[0, 0])
    y_point = np.mean(contours_ARRAY[0].T[1, 0])
    return x_point, y_point


# SELECT-FILE=========================================================================
movie = cv2.VideoCapture(file_path)

# MOVIE-PROFILE=========================================================================
fps = int(movie.get(cv2.CAP_PROP_FPS))
width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # four-caracter-code
video = cv2.VideoWriter(output_filename, fourcc, fps, (width, height), True)


# DEBUGE================================================================================
if not movie.isOpened():  # if you cannot open the movie file, kill this process.
    sys.exit()

while True:
    ret, frame = movie.read()

    if ret:

        frame1 = cv2.resize(frame, dsize=(600, 400))

        # change BGR to HSV
        # H: 0-179, S: 0-255, V: 0-255
        img_HSV = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)  # Gray scale
        # smoothing
        img_HSV = cv2.GaussianBlur(img_HSV, (5, 5), 0)

        '''
        #split into 3 channels
        img_H, img_S, img_V = cv2.split(img_HSV) 

        ret, img_thresh1 = cv2.threshold(
            img_H, 0, 255, cv2.THRESH_BINARY_INV)
        ret, img_thresh2 = cv2.threshold(
            img_H, 10 / 360 * 179, 255, cv2.THRESH_BINARY_INV)

        img_thresh = 255 - (img_thresh2 - img_thresh1)
        '''

        # 赤色のHSVの値域1
        hsv_min = np.array([0, 64, 0])
        hsv_max = np.array([30, 255, 255])
        mask1 = cv2.inRange(img_HSV, hsv_min, hsv_max)

        # 赤色のHSVの値域2
        hsv_min = np.array([150, 64, 0])
        hsv_max = np.array([179, 255, 255])
        mask2 = cv2.inRange(img_HSV, hsv_min, hsv_max)

        # 赤色領域のマスク（255：赤色、0：赤色以外）
        mask = mask1 + mask2

        # マスキング処理
        img_thresh = cv2.bitwise_and(frame1, frame1, mask=mask)
    cv2.imshow('default', frame1)
    cv2.imshow(window_name, img_thresh)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    else:
        movie.set(cv2.CAP_PROP_POS_FRAMES, 0)
cv2.destroyWindow(window_name)
cv2.destroyWindow('default')


# MAIN==================================================================================
x_list = []
y_list = []

while True:
    ret, frame = movie.read()  # while文中なので必要

    if not ret:
        break

    x, y = Find_contours(frame)

    frame = cv2.circle(frame, (int(x), int(y)), 30, (0, 255, 0), 3)

    video.write(frame)
    x_list.append(x)
    y_list.append(y)

movie.release()


print("All Process Finished!")
cv2.waitKey()
