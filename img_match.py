import cv2
import numpy as np
import copy
import os
import matplotlib.pyplot as plt


# ref: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
def kp_des_match(img_std, img_cur):

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img_std, None)
    kp2, des2 = orb.detectAndCompute(img_cur, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)
    distances = [matches[index].distance for index in range(10)]
    d_sum = np.sum(distances)
    d_avg = np.average(distances)

    # uncomment the following to show every single matching

    # img_res = cv2.drawMatches(img_std, kp1, img_cur, kp2, matches[:10], None, flags = 2)

    # plt.imshow(img_res)
    # plt.show()

    print('Comparison part is done. Matches distance are:\n')
    print(distances, '\nSum: ', d_sum, '\nAvg:', d_avg)
    print('\n')

    return d_avg # average distance of matching points, the lower the better


if __name__ == "__main__":
    
    cur_path = os.getcwd() + '\\data'
    std_path = cur_path + '\\std\\0.png' # standard image(position)
    cmp_path = cur_path + '\\unlabeled_images'

    g_path = cmp_path + '\\good_pos' # images with 'good' position
    b_path = cmp_path + '\\bad_pos' # images with 'bad' position
    
    img_std = cv2.imread(std_path, 0)
    
    imgs_cmp_g = os.listdir(g_path)
    imgs_cmp_b = os.listdir(b_path)

    g_d_avg = []
    b_d_avg = []

    for img_cmp_single in imgs_cmp_g:

        img_path = os.path.join(g_path, img_cmp_single)
        img_cur = cv2.equalizeHist(cv2.imread(img_path, 0))

        try:
            result = kp_des_match(img_std, img_cur)

        except TypeError:
            print('Unable to find enough matches. Proceed to next image.\n')
            g_d_avg.append(0)

        else:
            g_d_avg.append(result)

    for img_cmp_single in imgs_cmp_b:

        img_path = os.path.join(b_path, img_cmp_single)
        img_cur = cv2.equalizeHist(cv2.imread(img_path, 0))

        try:
            result = kp_des_match(img_std, img_cur)

        except TypeError:
            print('Unable to find enough matches. Proceed to next image.\n')
            b_d_avg.append(0)

        else:
            b_d_avg.append(result)

plt.plot(range(len(g_d_avg)), g_d_avg)
plt.plot(range(len(b_d_avg)), b_d_avg)
plt.show()
# no obvious difference detected
# thus this method's of not much usefulness

print('The end')
