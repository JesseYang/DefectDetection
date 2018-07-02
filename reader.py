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
    def __init__(self, filename_list, flip_ver=True, flip_horiz=True, shuffle=True):
        super(Data, self).__init__()

        content = []
        for filename in filename_list:
            with open(filename) as f:
                content.extend(f.readlines())

        self.imglist = [x.strip() for x in content] 

        self.flip_ver = flip_ver
        self.flip_horiz = flip_horiz
        self.shuffle = shuffle

    def size(self):
        return len(self.imglist)

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.imglist)

        for img_path in self.imglist:

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # wld
            if cfg.wld == True:
                img = wld(img)

            # random scale
            scale = random.choice(cfg.scale_list)
            if scale >= 2:
                img = cv2.pyrDown(img)
            if scale >= 4:
                img = cv2.pyrDown(img)
            img = np.expand_dims(img, axis=-1)

            # random flip
            if self.flip_ver == True and np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

            if self.flip_horiz == True and np.random.rand() > 0.5:
                img = cv2.flip(img, 0)

            # random crop
            h, w, _ = img.shape
            assert h >= cfg.img_size and w >= cfg.img_size, (h, w)
            diffh = h - cfg.img_size
            h0 = 0 if diffh == 0 else np.random.randint(diffh)
            diffw = w - cfg.img_size
            w0 = 0 if diffw == 0 else np.random.randint(diffw)

            img = img[h0:h0 + cfg.img_size, w0:w0 + cfg.img_size]

            yield [img, np.copy(img)]

    def reset_state(self):
        super(Data, self).reset_state()


if __name__ == '__main__':
    ds = Data(['train_good_2.txt'], True)

    ds.reset_state()
    
    g = ds.get_data()
    sample = next(g)
