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

def predict_image(input_path, output_path, predict_func):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if cfg.wld:
        img = wld(img)

    diff_list = []
    for scale in cfg.inf_scales:
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

        if scale >= 2:
            diff = cv2.pyrUp(diff)
        if scale >= 4:
            diff = cv2.pyrUp(diff)

        diff_list.append(diff)

        misc.imsave('diff_%d.jpg' % scale, diff)


    f = open('th_info.pkl', 'rb')
    th_info = pickle.load(f)

    output_list = []
    for scale_idx, scale in enumerate(cfg.inf_scales):
        info = th_info[scale]
        diff = diff_list[scale_idx]
        th = info['mean'] + 3 * info['std']
        layer_output = (diff > th).astype(np.int)
        output_list.append(layer_output)
        
    output_1 = output_list[0] & output_list[1]
    output_2 = output_list[1] & output_list[2]
    output = output_1 | output_2

    misc.imsave('output.jpg', output)

    '''
    diff = np.sum(diff_list, axis=0)
    misc.imsave('diff.jpg', diff)

    diff_blur = cv2.blur(diff, (5, 5))
    misc.imsave('diff_blur.jpg', diff_blur)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--input_path', help='path of the input image')
    parser.add_argument('--output_path', help='path of the output image', default='output.png')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    predict_func = get_pred_func(args)
    
    predict_image(args.input_path, args.output_path, predict_func)
