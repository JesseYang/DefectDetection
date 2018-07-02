#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
except Exception:
    from cfgs.config import cfg

try:
    from .train import Model
except Exception:
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
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    image = image[0:cfg.img_size, 0:cfg.img_size]
    image = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)

    predictions = predict_func(image)

    img_input = predictions[0]
    img_pred = predictions[1]

    misc.imsave('1.jpg', img_input[0,:,:,0])
    misc.imsave('2.jpg', img_pred[0,:,:,0])
    misc.imsave('diff.jpg', np.abs(img_input[0,:,:,0] - img_pred[0,:,:,0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path of the model waiting for validation.')
    parser.add_argument('--input_path', help='path of the input image')
    parser.add_argument('--output_path', help='path of the output image', default='output.png')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    predict_func = get_pred_func(args)
    
    predict_image(args.input_path, args.output_path, predict_func)
