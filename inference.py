import os
import cv2
import pickle
import glob
import random
import imutils
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import argparse
from config import NUM_CLASS

NUM_TTA = 1000

cls_to_idx = {'Deng': 0, 'Dengb': 1, 'Dengl': 2, 'SimSun': 3, 'msyh': 4, 'msyhbd': 5, 'simfang': 6, 'simhei': 7, 'simkai': 8}
idx_to_cls = {v: k for k, v in cls_to_idx.items()}
print(idx_to_cls)

def convert_to_even(x):
    return 2 * ((int(x) + 1) // 2)

def process_image(img, target=224, prior_size=256, center_crop=True):
    h, w = img.shape[:2]
    if h > w:
        img = cv2.resize(img, (prior_size, convert_to_even(prior_size * h / w))) / 255.0
    else:
        img = cv2.resize(img, (convert_to_even(prior_size * w / h), prior_size)) / 255.0
    # print('bar: ', img.shape)
    h, w = img.shape[:2]
    if center_crop:
        PAD_H, PAD_W = (h - target) // 2, (w - target) // 2
        img = img[PAD_H:-PAD_H, PAD_W:-PAD_W, ...]
    else:
        PAD_H, PAD_W = h - target, w - target
        pad_top = random.randrange(PAD_H)
        pad_down = PAD_H - pad_top
        pad_left = random.randrange(PAD_W)
        pad_right = PAD_W - pad_left
        img = img[pad_top:-pad_down, pad_left:-pad_right, ...]
    # print('foo: ', img.shape)

    img = np.transpose(img, [2,0,1])
    img = np.reshape(img, [1,3,target,target]).astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for c in range(3):
        img[:,c,...] = (img[:,c,...] - mean[c]) / std[c]
    img = torch.from_numpy(img)
    return img

class FontClassifier():
    def __init__(self, model_path='output/models/checkpoint/resnet34-epoch-1.pth',
                 log_file='output/result.log', crop_size=224, input_size=256):
        self.model_path = model_path
        self.crop_size = crop_size
        self.input_size = input_size
        self.log_file = log_file

        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def inference(self, img_path):
        assert os.path.exists(self.model_path)
        saved_model = torch.load(self.model_path, 'cpu')
        model = models.resnet34()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASS)
        model.load_state_dict(saved_model)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        im = cv2.imread(img_path)
        print("img_path: ", img_path)
        print("input image: ", im.shape)

        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        count = np.zeros(NUM_CLASS)
        for i in range(NUM_TTA):
            ## image process
            img_i = process_image(im, self.crop_size, self.input_size, center_crop=False)
            # print('image process: ', img_i.shape)

            ## inference
            with torch.no_grad():
                if torch.cuda.is_available():
                    img_i = img_i.cuda()
                img_i = Variable(img_i)

                output = model(img_i)
                _, pred = torch.max(output, 1)
                prob = torch.nn.functional.softmax(output, dim=1)

                # print("prob: ", prob)
                # prob_good = p.float().squeeze().tolist()[1]
                # prob_list.append(prob_good)
                # if prob_good > max_prob:
                #     max_prob = prob_good
                #     max_idx = idx
                # pred = pred.int().tolist()[0]
                # print("pred: ", pred)
                assert pred >= 0 and pred <= NUM_CLASS-1
                count[pred] += 1
                # Draw detected blobs as red circles.
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
                # if pred == 1:
                #     color = (0, 255, 0)
                # else:
                #     color = (0, 0, 255)
                # im = cv2.drawKeypoints(im, [kp], np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        result = idx_to_cls[np.argmax(count)]
        print('count: ', count)
        print('result: ', result)
        print('-' * 100)
        # # visualize result
        # im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

        # for idx, kp in enumerate(keypoints):
        #     # if idx == max_idx:
        #     if prob_list[idx] > THRESH:
        #         color = (0, 255, 0)
        #     else:
        #         color = (0, 0, 255)
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     point = tuple(map(int, kp.pt))
        #     fontScale = 1
        #     fontColor = color
        #     lineType = 2

        #     cv2.putText(im_bgr, '%f' %prob_list[idx], point, font, fontScale, fontColor, lineType)

        #     if self.enclosing_type == EnclosureType.RECT:
        #         box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(enclosure_list[idx])
        #         box = np.int0(box - 0.5 * self.patch_size)
        #         cv2.drawContours(im_bgr, [box + np.int0(np.array(kp.pt))], 0, color, lineType)
        #     elif self.enclosing_type == EnclosureType.ECLLIPSE:
        #         ellipse = enclosure_list[idx]
        #         ellipse2 = list(ellipse)
        #         ellipse2[0] = kp.pt
        #         cv2.ellipse(im_bgr, tuple(ellipse2), color, lineType)

        #     im = cv2.drawKeypoints(im_bgr, [kp], np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # # Show keypoints
        # #cv2.imshow("Keypoints", im_bgr)
        # cv2.imwrite(os.path.join("../figure", 'dl_' + os.path.basename(img_path)), im_bgr)
        # #cv2.waitKey(0)

        # save result to log file
        with open(self.log_file, 'a') as f:
            f.write(img_path + '\n')
            f.write(result + '\n')
            # classify images into folders
            out_dir = os.path.join(os.path.dirname(self.log_file), 'times', result)
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            img_name = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(out_dir, img_name))
            
            

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--file_type", type=str, default="file", help="file/folder")
    # parser.add_argument("--path", type=str, default="D:/ai_medi_test_img/2019_03_12_13_46_31.jpg", help="file/folder path")
    # args = parser.parse_args()

    font_classifier = FontClassifier(model_path='output/models/checkpoint/resnet34-epoch-12.pth', log_file='output/result.log',
                          crop_size=112, input_size=128)

    # in_dir = 'input/test/simhei'
    in_dir = 'test_images2'

    if os.path.isdir(in_dir):
        # images = glob.glob(os.path.join(in_dir, '**', '*.jpg'), recursive=True)[:20]        
        images = glob.glob(os.path.join(in_dir, '**', '*.jpg'), recursive=True)
        for img in tqdm(images, desc='running ...', ncols=100):
            font_classifier.inference(img)