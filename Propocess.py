# -*- coding: utf-8 -*-

import numpy as np
import cv2
import math
import imutils
import linear_regression
from matplotlib import pyplot as plt 
from image import Image


def imshow_2(pic1, pic2, figure_name='Figure', name1='pic1', name2='pic2'):
    # %matplotlib qt5
    # show the result side by side 
    plt.figure(figure_name)
    plt.subplot(2, 2, 1)
    plt.imshow(pic1)
    plt.title(name1)
    plt.subplot(2, 2, 2)
    plt.imshow(pic2)
    plt.title(name2)
    plt.show()


def FindGround(image):
    image.histogram_equalization()
    image.gamma_trans_gray(gamma=0.2)
    mask, _ = image.edge_canny(thr_min=50, thr_max=100, thresh=150, struc=5, remove_threshold=400)
    
    # show the result side by side    
    img_RGB = cv2.cvtColor(image.img, cv2.COLOR_YUV2RGB)
    imshow_2(pic1=img_RGB, pic2=mask)


def FindM(image):
    image.histogram_equalization()
    image.gamma_trans_gray(gamma=0.4)
    # image.laplace_sharpen()
    try:
        mask, _ = image.edge_canny(thr_min=50, thr_max=100, thresh=150, struc=5, remove_threshold=400)
        horizon = cv2.resize(mask, (256, 256))
        slope = linear_regression.run(horizon)
        angle = - (math.atan(slope) / math.pi) * 180
        image.img = imutils.rotate_bound(image.img, angle)
        # show the result side by side
        img_RGB = cv2.cvtColor(image.img, cv2.COLOR_YUV2RGB)
        imshow_2(pic1=img_RGB, pic2=mask)
    except:
        print('The information that can be extracted is limited !')


image = Image()
image_name = '../Dataset/other.jpg'
img_BGR = cv2.imread(image_name) # img is BGR
img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV) # img is YUV
image.open(img_YUV)
# FindGround(image)
FindM(image)
# imshow_2(pic1=img_RGB, pic2=img_LAB)