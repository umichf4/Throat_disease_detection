# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:53:05 2019

Process: EqualizeHist,Gamma,Canny

@author: Pi
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt 
from image import Image

def imshow_2(pic1, pic2, figure_name='Figure', name1='pic1', name2='pic2'):
    # %matplotlib qt5 
    # show the result side by side 
    plt.figure(figure_name)
    plt.subplot(2,2,1)
    plt.imshow(pic1)
    plt.title(name1)
    plt.subplot(2,2,2)
    plt.imshow(pic2)
    plt.title(name2)
    plt.show()
    
def FindGround(image):
    img_hist = image.histogram_equalization()
    img_gamma = image.gamma_trans_gray(gamma=0.2)
    mask, _ = image.edge_canny(thr_min=50, thr_max=100, thresh=150, struc=5, remove_threshold=400)
    
    # show the result side by side    
    img_RGB = cv2.cvtColor(image.img, cv2.COLOR_YUV2RGB)
    imshow_2(pic1=img_RGB, pic2=mask)

def FindM(image):
    img_hist = image.histogram_equalization()
    img_gamma = image.gamma_trans_gray(gamma=0.2)
    mask, _ = image.edge_canny(thr_min=50, thr_max=100, thresh=150, struc=5, remove_threshold=400)
    
    # show the result side by side    
    img_RGB = cv2.cvtColor(image.img, cv2.COLOR_YUV2RGB)
    imshow_2(pic1=img_RGB, pic2=mask)
    
image = Image()
image_name = '../Dataset/Test.jpg'
img_BGR = cv2.imread(image_name) # img is BGR
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV) # img is YUV
image.open(img_YUV)
#FindGround(image)
FindM(image)