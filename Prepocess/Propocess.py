# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:53:05 2019

@author: Pi
"""
import numpy as np
import cv2

img = cv2.imread('../Dataset/Test.jpg') # JPG is yuv
img_y = img[:,:,0] # gray image
cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.imshow("img", img)
cv2.waitKey(0)

img_channels = cv2.split(img)
img_channels[0] = cv2.equalizeHist(img_channels[0])

img = cv2.merge(img_channels)
cv2.namedWindow('img after equalizeHist', cv2.WINDOW_KEEPRATIO)
cv2.imshow("img after equalizeHist", img)
cv2.waitKey(0)
