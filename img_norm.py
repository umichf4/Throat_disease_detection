import os
import cv2
import numpy as np


def rgb_norm(img_ori): # color correction
    
    b, g, r = cv2.split(img_ori)

    sum = b.astype(int) + g.astype(int) + r.astype(int)
    b_new = np.uint8(b / sum * 255)
    g_new = np.uint8(g / sum * 255)
    r_new = np.uint8(r / sum * 255)
    
    img_alt = cv2.merge((b_new, g_new, r_new))

    return img_alt


def rgb_histeq(img_ori): # perform histogram equalization in rgb channels respectively

    b, g, r = cv2.split(img_ori)

    b_new = cv2.equalizeHist(b)
    g_new = cv2.equalizeHist(g)
    r_new = cv2.equalizeHist(r)
    
    img_alt = cv2.merge((b_new, g_new, r_new))

    return img_alt


def yuv_histeq(img_ori): # perform histogram equalization in y channel

    img_yuv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YUV)

    y, u, v = cv2.split(img_yuv)

    y_new = cv2.equalizeHist(y)
    
    img_alt = cv2.merge((y_new, u, v))

    return cv2.cvtColor(img_alt, cv2.COLOR_YUV2BGR)


def ybr_histeq(img_ori): # perform histogram equalization in y channel

    img_ycrcb = cv2.cvtColor(img_ori, cv2.COLOR_BGR2YCrCb)

    y, Cr, Cb = cv2.split(img_ycrcb)

    y_new = cv2.equalizeHist(y)
    
    img_alt = cv2.merge((y_new, Cr, Cb))

    return cv2.cvtColor(img_alt, cv2.COLOR_YCrCb2BGR)


def hsv_histeq(img_ori): # perform histogram equalization in v channel

    img_hsv = cv2.cvtColor(img_ori, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_hsv)

    v_new = cv2.equalizeHist(v)
    
    img_alt = cv2.merge((h, s, v_new))

    return cv2.cvtColor(img_alt, cv2.COLOR_HSV2BGR)


if __name__ == '__main__':

    cur_path = os.getcwd() + '\\data\\test'
    imgs = os.listdir(cur_path)

    for img_single in imgs:

        # histeq in y channel (yuv or ycbcr) is the best
        # color correction after histeq seems to be the best
        
        img_path = os.path.join(cur_path, img_single)
        img_ori = cv2.imread(img_path)
        img_corrected_rgb = rgb_norm(img_ori)
        img_hist_yuv = yuv_histeq(img_ori)
        img_corrected_hist_yuv = yuv_histeq(img_corrected_rgb)
        img_hist_corrected_yuv = rgb_norm(img_hist_yuv)

        # img_lab = cv2.cvtColor(img_ori, cv2.COLOR_BGR2Lab)
        # img_show = cv2.cvtColor(img_lab, cv2.COLOR_lab2)

        # gray_1 = cv2.cvtColor(img_hist_yuv, cv2.COLOR_RGB2GRAY)
        # gray_2 = cv2.cvtColor(img_hist_corrected_yuv, cv2.COLOR_RGB2GRAY)
        # gray_all = np.hstack((gray_1, gray_2))

        img_row_1 = np.hstack((img_ori, img_hist_yuv))
        img_row_2 = np.hstack((img_corrected_hist_yuv, img_hist_corrected_yuv))
        img_all = np.vstack((img_row_1, img_row_2))

        cv2.imwrite(img_single + '_corrected_hist.jpg', img_hist_corrected_yuv)

        cv2.namedWindow(img_single, cv2.WINDOW_FREERATIO)
        cv2.imshow(img_single, img_all)
        cv2.waitKey(0)
