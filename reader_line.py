import os
import random
import sys

from tensorpack import *
import numpy as np
from scipy import misc
import cv2
import json
from cfgs.config import cfg
import time
from scipy import signal

def wld(img):
    filter1 = np.array([[-1,-1,-1],
                        [-1, 8,-1],
                        [-1,-1,-1]])
    filter2 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]])

    temp1 = signal.convolve2d(img, filter1, boundary='symm', mode='same')
    temp2 = signal.convolve2d(img, filter2, boundary='symm', mode='same')

    ret = np.arctan(temp1 / (temp2 + 1e-5))

    max_val = np.arctan(np.inf)
    min_val = np.arctan(-np.inf)

    ret = (ret - min_val) / (max_val - min_val) * 255

    return ret

class Data(RNGDataFlow):
    def __init__(self, filename_list, rotate=True, flip_ver=True, flip_horiz=True, shuffle=True, line_noise=False):
        super(Data, self).__init__()

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.imglist = [x.strip() for x in content] 

        self.flip_ver = flip_ver
        self.flip_horiz = flip_horiz
        self.rotate = rotate
        self.shuffle = shuffle
        self.line_noise = line_noise

    def size(self):
        return len(self.imglist)

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.imglist)

        for img_path in self.imglist:

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#             label = np.copy(img)
#             if self.line_noise:
#                 xmin = np.random.randint(0, img.shape[1] - 110)
#                 ymin = np.random.randint(0, img.shape[0] - 110)
#                 xlength = np.random.randint(90, 110)
#                 ylength = np.random.randint(0, xlength)
#                 cv2.line(img,(xmin,ymin),(xmin+xlength,ymin+ylength), 100, 3)

            # wld
            if cfg.wld == True:
                img = wld(img)

            # random scale
            line_noise_length = np.random.randint(90, 110)
            scale = random.choice(cfg.scale_list)
            if scale >= 2:
                img = cv2.pyrDown(img)
                line_noise_length = line_noise_length // 2
            if scale >= 4:
                img = cv2.pyrDown(img)
                line_noise_length = line_noise_length // 2

            # random rotate
            angle = random.choice([0, 90, 180, 270])
            if self.rotate and angle != 0:
                rows, cols = img.shape
                M = cv2.getRotationMatrix2D((cols / 2,rows / 2), angle, 1)
                img = cv2.warpAffine(img, M, (cols, rows))

            # random flip
            if self.flip_ver == True and np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

            if self.flip_horiz == True and np.random.rand() > 0.5:
                img = cv2.flip(img, 0)

            img = np.expand_dims(img, axis=-1)
#             label = np.expand_dims(label, axis=-1)


            # random crop
            h, w, _ = img.shape
            assert h >= cfg.img_size and w >= cfg.img_size, (h, w)
            diffh = h - cfg.img_size
            h0 = 0 if diffh == 0 else np.random.randint(diffh)
            diffw = w - cfg.img_size
            w0 = 0 if diffw == 0 else np.random.randint(diffw)

            img = img[h0:h0 + cfg.img_size, w0:w0 + cfg.img_size]

            label = np.copy(img)
            if self.line_noise:
                xmin = np.random.randint(0, img.shape[1])
                ymin = np.random.randint(0, img.shape[0])
#                 import pdb
#                 pdb.set_trace()
                ylength = np.random.randint(0, line_noise_length)
                cv2.line(img,(xmin,ymin),(xmin+line_noise_length,ymin+ylength), 100, 4)

            yield [img, label]

    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    ds = Data(['train_good_10.txt'], shuffle=False, line_noise=True)

    ds.reset_state()
    
    g = ds.get_data()
    for i in range(100):
        img, label = next(g)
        cv2.imwrite('line/%d.png'%i, img)
