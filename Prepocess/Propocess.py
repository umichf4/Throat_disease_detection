# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:53:05 2019

Process: EqualizeHist and Gamma

@author: Pi
"""
import numpy as np
import cv2
from Gamma import gamma_trans
from image import Image

#from image import AWB
image = Image()
image_name = '../Dataset/Test.jpg'
img_BGR = cv2.imread(image_name) # img is BGR
img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV) # img is YUV
image.open(img)
#img_y = image.img[:,:,0] # gray image
#cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
#cv2.imshow("img", img_BGR)
#cv2.waitKey(0)

img_channels = cv2.split(image.img)
img_channels[0] = cv2.equalizeHist(img_channels[0])
img_channels[0] = gamma_trans(img_channels[0], 0.1)
image.img = cv2.merge(img_channels)
image.img = cv2.cvtColor(image.img, cv2.COLOR_YUV2BGR)
mask, _ = image.edge_canny()

#cv2.namedWindow('img after equalizeHist', cv2.WINDOW_KEEPRATIO)
#cv2.imshow("img after equalizeHist", img)
#cv2.waitKey(0)

#horizontal_stack = np.hstack((image.img, image.img))
horizontal_stack = np.hstack((img[:,:,0], mask))
cv2.namedWindow('Before & After', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Before & After", horizontal_stack)
cv2.waitKey(0)

#img= cv2.resize(img, (25, 25))
#img = ace(img)
#cv2.namedWindow('img after ACE', cv2.WINDOW_KEEPRATIO)
#cv2.imshow("img after ACE", img)
#cv2.waitKey(0)
