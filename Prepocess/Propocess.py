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
    #img_RGB = cv2.cvtColor(image.img, cv2.COLOR_YUV2RGB)
    #imshow_2(pic1=img_RGB, pic2=mask)
    
    return mask

def FindM(image, mask):
    # Find the inside area
    cor = np.where(mask)
    cor_h = cor[0]
    cor_w = cor[1]
    cor_up = cor_h.min()
    cor_down = cor_h.max()
    cor_left = cor_w.min()
    cor_right = cor_w.max()
    Ground_len = cor_right - cor_left
    area = [cor_down-Ground_len//2, cor_down, cor_left, cor_right] # Up Down Left Right
    img_inside = image.img[area[0]:area[1], area[2]:area[3],:]
    img_inside_RGB = cv2.cvtColor(img_inside, cv2.COLOR_YUV2RGB)
    #imshow_1(pic=img_inside_RGB)
    mask_global = mask
    image_inside = Image()
    image_inside.open(img_inside)
    img_gamma = image_inside.gamma_trans_gray(gamma=3)
    mask, _ = image_inside.edge_canny(thr_min=50, thr_max=200, thresh=255, struc=4, remove_threshold=100)
    
    # show the result side by side    
    img_RGB = cv2.cvtColor(image_inside.img, cv2.COLOR_YUV2RGB)
    imshow_2(pic1=img_RGB, pic2=mask)

    
image = Image()
image_name = '../Dataset/Test.jpg'
img_BGR = cv2.imread(image_name) # img is BGR
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV) # img is YUV
image.open(img_YUV)
mask = FindGround(image)
FindM(image, mask)