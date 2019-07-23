import cv2
import numpy as np
import copy
import os


def extract_feature(dst_img, ref_wh, feature_type):
    """
    Extract feature from the dst_img's ROI of rect.
    return: A vector of features value
    """
    if dst_img.shape != ref_wh:
        roi = cv2.resize(dst_img, (ref_wh[0], ref_wh[1]))

    else:
        roi = dst_img
    
    if feature_type == 'intensity':

        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        elif len(roi.shape) == 2:
            gray_roi = roi
        
        return gray_roi.astype(np.float).reshape(1, -1)/255.0

    elif feature_type == 'hog':
        hog = cv2.HOGDescriptor()
        scaled_roi = cv2.resize(roi, (64, 128))
        feature = hog.compute(scaled_roi)
        return feature


def compute_similarity(template, feature, distance_type):
    """
    Compute similarity of a single feature with template
    """

    if distance_type == 'cos_dis':
        #cosine distance
        feature_norm = feature / np.max(feature)
        template_norm = template / np.max(template)

        cos_theta = np.sum(feature_norm * template_norm) / np.sqrt(np.sum(feature_norm * feature_norm) * np.sum(template_norm * template_norm))

        return float(cos_theta)

    elif distance_type == 'euc_dis':
        # L2norm, that is Euclidean distance
        feature_norm = feature / np.max(feature)
        template_norm = template / np.max(template)

        dvalue = feature_norm - template_norm
        l2d = float(np.sqrt(np.sum(dvalue * dvalue)))
    
        return l2d

    elif distance_type == 'man_dis':
        # Manhattan distance
        feature_norm = feature / np.max(feature)
        template_norm = template / np.max(template)

        dif = [abs(feature_norm[index] - template_norm[index]) for index in range(len(feature))]
        
        man_dis = sum(list(dif[0]))

        return float(man_dis)


def img_compare(img_1, img_2, threshold, feature_type, distance_type):

    ref_wh = img_1.shape

    feature_1 = extract_feature(img_1, ref_wh, feature_type)
    feature_2 = extract_feature(img_2, ref_wh, feature_type)

    similarity = compute_similarity(feature_1, feature_2, distance_type)

    if distance_type == 'cos_dis':

        if similarity >= threshold:
            return [True, similarity]

        else:
            return [False, similarity]

    else:

        if similarity <= threshold:
            return [True, similarity]

        else:
            return [False, similarity]


if __name__ == "__main__":
    
    cur_path = os.getcwd() + '\\data\\after_norm'
    imgs = os.listdir(cur_path)
    img_0_path = os.path.join(cur_path, imgs[0])

    similarities = []
    
    for img_single in imgs:

        img_path = os.path.join(cur_path, img_single)
        img_cur = cv2.imread(img_path)
        img_0 = cv2.imread(img_0_path)

        try:
            similarity = img_compare(img_0, img_cur, 100, 'intensity' , 'man_dis')

        except TypeError:
            print('Undefined operation! \n')

        else:
            similarities.append(similarity[1])
            print(similarity[0], ' ', similarity[1], '\n')

print('The end')
