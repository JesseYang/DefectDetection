#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle
import os
import uuid
import shutil
import ntpath
import numpy as np
from scipy import misc
import argparse
import json
import cv2

import matplotlib.pyplot as plt

from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorpack import *

try:
    from .cfgs.config import cfg
    from .reader import wld
    from .train import Model
except Exception:
    from cfgs.config import cfg
    from reader import wld
    from train import Model

def get_pred_func(args):
    sess_init = SaverRestore(args.model_path)
    predict_config = PredictConfig(session_init=sess_init,
                                   model=Model(),
                                   input_names=["img_input"],
                                   output_names=["network_input", "img_pred"])

    predict_func = OfflinePredictor(predict_config)
    return predict_func

def calculate_th(predict_func):

    img_paths = []
    for train_file in cfg.train_list:
        f = open(train_file, 'r')
        img_paths += [e.strip() for e in f.readlines()]

    th_info = {}
    for scale in cfg.inf_scales:

        diff_list = []
        for img_path in img_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # wld
            if cfg.wld:
                img = wld(img)

            img_scale = np.copy(img)
            if scale >= 2:
                img_scale = cv2.pyrDown(img_scale)
            if scale >= 4:
                img_scale = cv2.pyrDown(img_scale)

            h, w = img_scale.shape[:2]
            diff = np.zeros((h, w))
    
            cur_h = 0
            while cur_h < h - cfg.overlap:
                end_h = cur_h + cfg.img_size - cfg.overlap
                start_h = cur_h - cfg.overlap
    
                if start_h < 0:
                    end_h = end_h - start_h
                    start_h = 0
    
                if end_h > h:
                    start_h = start_h - (end_h - h)
                    end_h = h
    
                cur_w = 0
                while cur_w < w - cfg.overlap:
                    end_w = cur_w + cfg.img_size - cfg.overlap
                    start_w = cur_w - cfg.overlap
    
                    if start_w < 0:
                        end_w = end_w - start_w
                        start_w = 0
    
                    if end_w > w:
                        start_w = start_w - (end_w - w)
                        end_w = w
    
                    sub_img = img_scale[start_h:end_h, start_w:end_w]
    
    
                    sub_img = np.expand_dims(np.expand_dims(sub_img, axis=-1), axis=0)
            
                    predictions = predict_func(sub_img)
                    sub_img_input = predictions[0]
                    sub_img_pred = predictions[1]
                    sub_img_diff = np.abs(sub_img_input[0,:,:,0] - sub_img_pred[0,:,:,0])
    
                    diff[start_h+cfg.overlap:end_h-cfg.overlap,start_w+cfg.overlap:end_w-cfg.overlap] = \
                        sub_img_diff[cfg.overlap:-cfg.overlap,cfg.overlap:-cfg.overlap]
    
                    cur_w = end_w - cfg.overlap
    
                cur_h = end_h - cfg.overlap

            real_diff = diff[cfg.overlap:-cfg.overlap, cfg.overlap:-cfg.overlap]
            diff_list.append(real_diff)

        diff_ary = np.asarray(diff_list)
        mean = np.mean(diff_ary)
        std = np.std(diff_ary)

        th_info[scale] = {'mean': mean, 'std': std}

    f = open('th_info.pkl', 'wb')
    pickle.dump(th_info, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    predict_func = get_pred_func(args)
    
    calculate_th(predict_func)
