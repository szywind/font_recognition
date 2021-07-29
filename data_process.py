import os
import math
# import csv
import glob
import cv2
import shutil
import pandas as pd
import numpy as np
import numpy.linalg as linalg
import imutils
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from enum import Enum


def get_square_roi_small(img, kp):
    H, W = img.shape[:2]
    x_left, y_top = kp.pt[:2]
    x_right = W - x_left
    y_bottom = H - y_top
    dx = min(x_left, x_right)
    dy = min(y_top, y_bottom)
    r = min(dx, dy)

    y1 = int(np.clip(y_top-r, 0, H))
    y2 = int(np.clip(y_top+r, 0, H))
    x1 = int(np.clip(x_left-r, 0, W))
    x2 = int(np.clip(x_left+r, 0, W))
    return img[y1:y2, x1:x2, ...]

def get_square_roi(img, kp, desired_size=160):
    H, W = img.shape[:2]
    x_left, y_top = kp.pt[:2]
    x_right = W - x_left
    y_bottom = H - y_top

    r = desired_size//2
    top = math.ceil(max(0, r-y_top))
    bottom = math.ceil(max(0, r-y_bottom))
    left = math.ceil(max(0, r-x_left))
    right = math.ceil(max(0, r-x_right))

    color = [255, 255, 255]
    print(top, bottom, left, right)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT,
                                value=color) # https://blog.csdn.net/ei1990/article/details/78350974
    H_aug, W_aug = img.shape[:2]

    y1 = int(np.clip(y_top+top-r, 0, H_aug))
    y2 = int(np.clip(y_top+top+r, 0, H_aug))
    x1 = int(np.clip(x_left+left-r, 0, W_aug))
    x2 = int(np.clip(x_left+left+r, 0, W_aug))

    roi = img[y1:y2, x1:x2, ...]
    assert(roi.shape[0] == desired_size and roi.shape[1] == desired_size)
    return roi

def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (linalg.norm(vec1) * linalg.norm(vec2))

def pt_inside(pt, box, epsilon=1e-6):
    vec = box - pt
    if len(box) < 2:
        return False
    orient = np.cross(vec[0], vec[1])
    for i in range(1, len(vec)-1):
        if abs(cos_sim(np.cross(vec[i], vec[i+1]), orient) - 1) > epsilon:
            return False
    return True


def get_min_area_enclosure(img, enclosure_type, debug=False):
    img_gray = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CROSS, element)

    img_gray = cv2.erode(img_gray, element, iterations=4)
    img_gray = cv2.dilate(img_gray, element, iterations=4)


    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU) # THRESH_OTSU cv2.THRESH_TRIANGLE
    if debug:
        cv2.imshow('Thresholded Image', thresh)
        cv2.waitKey(0)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnts = contours[0] if imutils.is_cv2() else contours[1]

    max_area = -1
    max_idx = -1
    for idx, cnt in enumerate(cnts):
        rect = cv2.minAreaRect(cnt)
        box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(rect)

        area = cv2.contourArea(cnt)
        if pt_inside((0.5*img.shape[1], 0.5*img.shape[0]), box) and area > max_area:
            max_area = area
            max_idx = idx

    if enclosure_type == EnclosureType.RECT:
        rect = cv2.minAreaRect(cnts[max_idx])
        if debug:
            print(rect, type(rect))
            box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(rect)
            box = np.int0(box)
            print(thresh.shape, box)
            cv2.drawContours(thresh, [box], 0, [255, 0, 0], 2)
            cv2.imshow("MinArea Rect", thresh)
            cv2.waitKey(0)

        return rect
    elif enclosure_type == EnclosureType.ECLLIPSE:
        ellipse = cv2.fitEllipse(cnts[max_idx])
        if debug:
            cv2.ellipse(thresh, ellipse, (255, 255, 0), 2)
            cv2.imshow("MinArea Ellipse", thresh)
            cv2.waitKey(0)

        return ellipse



def split_train_val_test(in_dir, out_dir, test_size=[0.1, 0.1], random_state=[20, 21]):
    classes = os.listdir(in_dir)
    
    for cls in tqdm(classes, desc='==> process classes ...', ncols=100):
        cls_data_list = glob.glob(os.path.join(in_dir, cls, '*.jpg'))

        train_set, test_set = train_test_split(cls_data_list, test_size=test_size[0], random_state=random_state[0])
        train_set, val_set = train_test_split(train_set, test_size=test_size[1], random_state=random_state[1])

        n_train = len(train_set)
        n_val = len(val_set)
        n_test = len(test_set)
        print(f'# train, # val, # test: {n_train}, {n_val}, {n_test}')

        for phase in ['train', 'val', 'test']:
            dst_dir = os.path.join(out_dir, phase, cls)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for src_img in tqdm(eval(f'{phase}_set'), desc='==> save data', ncols=100):
                # print(src_img)
                shutil.copy(src_img, dst_dir)

if __name__ == '__main__':
    split_train_val_test('../../data/0703_gen', 'input')
    
