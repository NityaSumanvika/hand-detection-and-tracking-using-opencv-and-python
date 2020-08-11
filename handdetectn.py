import cv2
import numpy as np
import imutils
import os
import copy
import math
import time

def calcfing(res,draw):
    hull = cv2.convexHull(res,returnPoints = False)
    if len(hull)>3:
        defects = cv2.convexityDefects(res,hull)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0]-start[0]) **2+ (end[1]-start[1]) **2)
                b = math.sqrt((far[0]-start[0]) **2+ (far[1]-start[1]) **2)
                c = math.sqrt((end[0]-far[0]) **2+ (end[1]-far[1]) **2)
                angle = math.acos((b ** 2 + c ** 2 - a**2)/(2 * b * c))
                if angle <=math.pi/2 :
                    cnt = cnt+1
                    cv2.circle(draw,far,8,[255,0,0],-1)
            if cnt>0:
                return True ,  cnt + 1
            else:
                return True , 0
    return False , 0
cam = cv2.VideoCapture(0)
cam.set(10, 200)

while cam.isOpened():
    ret,image = cam.read()
    image = cv2.bilateralFilter(image,5,50,100)
    image = cv2.flip(image,1)
    cv2.imshow('input',image)
    bgModel = cv2.createBackgroundSubtractorMOG2(0,50)#here 50 is sharpness
    fgmask = bgModel.apply(image)
    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.erode(fgmask,kernel)
    img = cv2.bitwise_and(image,image,mask = fgmask)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l = np.array([0,48,80] , dtype ="uint8")
    u = np.array([20,255,255] , dtype="uint8")
    skin = cv2.inRange(hsv, l ,u)
    cv2.imshow("hsv image", skin)

    hand = copy.deepcopy(skin)
    contours,hierarchy = cv2.findContours(hand,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    p=-1
    if length>0:
        for i in range(length):
            area = cv2.contourArea(contours[i])
            if area>p:
                p=area
                ci = i
                res = contours[ci]
        hull = cv2.convexHull(res)
        draw = np.zeros(img.shape , np.uint8)
        cv2.drawContours(draw, [res],0,(0,255,0),2)
        cv2.drawContours(draw, [hull], 0 , (0,225,0),3)
        isFinishCal, cnt = calcfing(res,draw)
        print ("fingers",cnt-1)
        time.sleep(1)
        cv2.imshow('output',draw)
        k = cv2.waitKey(10)
    if k == 27:
        break
