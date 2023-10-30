import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import joblib
import cv2
from config import *
from skimage import color
import matplotlib.pyplot as plt 
import os 
import glob

# 加载SVM模型
# clf = joblib.load("../data/modelsTest/svm.pkl")
cap = cv2.VideoCapture("F:/svm+hog/human-detector/video/bend.avi")

ret, frame1 = cap.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    
    # 计算帧间差分，获取运动区域
    diff = cv2.absdiff(gray1, gray2)
    
    thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
     # 循遍历每个轮廓
    for contour in contours:
        # 计算轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        print(x,y,w,h)

        # 可选：过滤小轮廓，以排除噪声
        if w > 20 and h > 20:
            # 在原始帧上绘制感兴趣的区域
            cv2.rectangle(gray1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示带有感兴趣区域的原始帧
    cv2.imshow('Video with ROIs', gray1)
    if cv2.waitKey(1) & 0xFF == 27:  # 按Esc键退出
        break
    


    # 提取感兴趣区域（ROI）并计算HOG特征
    # 进行HOG特征提取操作
    

    # # 使用SVM模型进行分类
    # predictions = model.predict(hog_features)

    # # 过滤掉不包含行人的区域
    # detected_regions = [roi for i, roi in enumerate(roi_list) if predictions[i] == 1]

    # # 在frame2上绘制检测结果
    # # 进行目标过滤和可视化操作

    # frame1 = frame2

# cap.release()
# cv2.destroyAllWindows()
