# track2.py
"""REFERENCE
https://qiita.com/fugunoko/items/7e5056449e172cbeadd9
https://hk29.hatenablog.jp/entry/2020/02/01/162533
https://algorithm.joho.info/programming/python/opencv-color-detection/
O'REILLY Open CV
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import subprocess
import os


file_path = 'MOV_1189.mp4'
output_filename = '220103-rover_detect-MOV_1189_revise.mp4'
window_name = 'frame'
delay = 1
abspath =  os.path.dirname(os.path.abspath(__file__))


def Find_contours(img):

    # change BGR to HSV
    # H: 0-179, S: 0-255, V: 0-255
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Gray scale
    # smoothing
    img_HSV = cv2.GaussianBlur(img_HSV, (5, 5), 0)
    #HS1
    hsv_min = np.array([0, 64, 0])
    hsv_max = np.array([30, 255, 255])
    mask1 = cv2.inRange(img_HSV, hsv_min, hsv_max)
    #HSV2
    hsv_min = np.array([150, 64, 0])
    hsv_max = np.array([179, 255, 255])
    mask2 = cv2.inRange(img_HSV, hsv_min, hsv_max)
    # 赤色領域のマスク（255：赤色、0：赤色以外）
    #mask = mask1 + mask2
    mask = mask2
    # マスキング処理
    img_thresh = cv2.bitwise_and(img, img, mask=mask)

    img_gray = cv2.cvtColor(img_thresh, cv2.COLOR_BGR2GRAY)  # Gray scale
    img_thresh = cv2.threshold(
        img_gray, 3, 255, cv2.THRESH_BINARY)[1]
        
    contours, hierarchy = cv2.findContours(
        img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find ROVER's contours

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
    ret, default_frame = movie.read()

    if ret:

        #frame_resize = cv2.resize(frame, dsize=(600, 400))
        img_crop = default_frame[0:720,400:1000]#[top:under,left:right]

        # change BGR to HSV
        # H: 0-179, S: 0-255, V: 0-255
        img_HSV = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)  # Gray scale
        # smoothing
        img_HSV = cv2.GaussianBlur(img_HSV, (5, 5), 0)

        #HSV1
        hsv_min = np.array([0, 64, 0])
        hsv_max = np.array([30, 255, 255])
        mask1 = cv2.inRange(img_HSV, hsv_min, hsv_max)

        #HSV2
        hsv_min = np.array([150, 64, 0])
        hsv_max = np.array([179, 255, 255])
        mask2 = cv2.inRange(img_HSV, hsv_min, hsv_max)

        # 赤色領域のマスク（255：赤色、0：赤色以外）
        #mask = mask1 + mask2
        mask = mask2

        # マスキング処理
        img_thresh = cv2.bitwise_and(img_crop, img_crop, mask=mask)
        img_gray = cv2.cvtColor(img_thresh, cv2.COLOR_BGR2GRAY)

        # デルタ画像を閾値処理を行う
        thresh = cv2.threshold(img_gray, 3, 255, cv2.THRESH_BINARY)[1]
        # 画像の閾値に輪郭線を入れる
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame = cv2.drawContours(img_crop, contours, -1, (0, 255, 0), 3) #all contours = -1

        cv2.imshow('default', default_frame)
        #cv2.imshow('crop',img_crop)
        cv2.imshow('panda', img_gray)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    else:
        movie.set(cv2.CAP_PROP_POS_FRAMES, 0)
cv2.destroyWindow(window_name)
cv2.destroyWindow('default')
cv2.destroyWindow('panda')
cv2.destroyWindow('crop')



# MAIN==================================================================================
x_list = []
y_list = []

movie = cv2.VideoCapture(file_path)

while True:
    ret, frame = movie.read()  # while文中なので必要

    if not ret:
        break

    img_crop = frame[0:720,400:1000]#[top:under,left:right]
    x, y = Find_contours(img_crop)

    frame = cv2.circle(frame, (int(x)+400, int(y)), 30, (0, 255, 0), 3)
    #frame = cv2.rectangle(frame, (int(x)+300, int(y)-50), (int(x)+500, int(y)+50), (0, 255, 0))

    video.write(frame)
    x_list.append(x)
    y_list.append(y)

movie.release()


print("All Process Finished!")
subprocess.Popen(["explorer", abspath], shell=True)
cv2.waitKey()
