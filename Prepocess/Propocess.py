# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:53:05 2019

Process: EqualizeHist,Gamma,Canny

@author: Pi
"""
import numpy as np
import cv2
from image import Image

image = Image()
image_name = '../Dataset/Test.jpg'
img_BGR = cv2.imread(image_name) # img is BGR
img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV) # img is YUV
image.open(img_YUV)

img_hist = image.histogram_equalization()
img_gamma = image.gamma_trans_gray(gamma=0.2)
mask, _ = image.edge_canny(thr_min=50, thr_max=100, thresh=150, struc=5, remove_threshold=400)

horizontal_stack = np.hstack((image.img[:,:,0], mask))
# show the result side by side
cv2.namedWindow('Before & After', cv2.WINDOW_KEEPRATIO)
cv2.imshow("Before & After", horizontal_stack)
cv2.waitKey(0)