# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:53:05 2019

@author: Pi
"""
import numpy as np
import cv2

img = cv2.imread('../Dataset/Test.jpg') # JPG is yuv
img_y = img[:,:,0]

cv2.namedWindow('demo', 0)
cv2.imshow("img_y", img)
cv2.waitKey(0)