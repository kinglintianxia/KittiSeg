#!/usr/bin/env python
# -*- coding: utf-8 -*-
# coding: utf-8

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
from seg_utils import seg_utils as seg

import tensorflow as tf
import time

import tensorvision
import tensorvision.utils as utils


def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0  # [0,1,2, ..., 255]/255.0 = [0.0, 1.0] float array.

    road_color = np.array(hypes['data']['road_color'])  # [255,0,255]
    background_color = np.array(hypes['data']['background_color'])  # [255,0,0], 'red'
    # np.all 会将比如说[384,1248,3] 变为[384,1248], 将3-D变为2-D。
    gt_road = np.all(gt_image == road_color, axis=2)        # gt road pixels, 'True' or 'False'.
    gt_bg = np.all(gt_image == background_color, axis=2)    # gt background pixels, 'True' or 'False'.
    valid_gt = gt_road + gt_bg  # 有效区域为True,无效区域也就是干扰（噪音）为False
    # FN, FP, TP+FN, FP+TN
    FN, FP, posNum, negNum = seg.evalExp(gt_road, cnn_image,    # seg.evalExp: 'submodules/evaluation/kitti_devkit/seg_utils.py'
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum 


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image

# 
def evaluate(hypes, sess, image_pl, inf_out):

    softmax = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']    # data_dir: 'DATA'

    eval_dict = {}
    for phase in ['train', 'val']:
        data_file = hypes['data']['{}_file'.format(phase)]  # "train_file": "data_road/train3.txt", "val_file": "data_road/val3.txt"
        data_file = os.path.join(data_dir, data_file) # data_file: 'DATA/data_road/train3.txt'
        # os.path.dirname: 去掉文件名，返回目录
        image_dir = os.path.dirname(data_file)   # image_dir: 'DATA/data_road/training/image_2'

        thresh = np.array(range(0, 256))/255.0
        total_fp = np.zeros(thresh.shape)   # FP
        total_fn = np.zeros(thresh.shape)   # FN
        total_posnum = 0    # TP
        total_negnum = 0    # TN

        image_list = []

        with open(data_file) as file:
            for i, datum in enumerate(file):
                    datum = datum.rstrip() # 删除 string 字符串末尾的指定字符（默认为空格)
                    image_file, gt_file = datum.split(" ") # train_image, gt_image
                    image_file = os.path.join(image_dir, image_file) # image_file: 'DATA/data_road/training/image_2/um_000000.png'
                    gt_file = os.path.join(image_dir, gt_file)  # gt_file: 'DATA/data_road/training/image_2/um_000000.png'

                    image = scp.misc.imread(image_file, mode='RGB')
                    gt_image = scp.misc.imread(gt_file, mode='RGB')

                    if hypes['jitter']['fix_shape']:    # Default: false
                        shape = image.shape
                        image_height = hypes['jitter']['image_height']
                        image_width = hypes['jitter']['image_width']
                        assert(image_height >= shape[0])
                        assert(image_width >= shape[1])

                        offset_x = (image_height - shape[0])//2
                        offset_y = (image_width - shape[1])//2
                        new_image = np.zeros([image_height, image_width, 3])
                        new_image[offset_x:offset_x+shape[0],
                                  offset_y:offset_y+shape[1]] = image
                        input_image = new_image
                    elif hypes['jitter']['reseize_image']:  # Default: false
                        image_height = hypes['jitter']['image_height']
                        image_width = hypes['jitter']['image_width']
                        gt_image_old = gt_image
                        image, gt_image = resize_label_image(image, gt_image,
                                                             image_height,
                                                             image_width)
                        input_image = image
                    else:
                        input_image = image

                    shape = input_image.shape

                    feed_dict = {image_pl: input_image}

                    output = sess.run([softmax], feed_dict=feed_dict)
                    # output[0]:image, output[1]: label
                    output_im = output[0][:, 1].reshape(shape[0], shape[1])

                    if hypes['jitter']['fix_shape']:    # Default: false
                        gt_shape = gt_image.shape
                        output_im = output_im[offset_x:offset_x+gt_shape[0],
                                              offset_y:offset_y+gt_shape[1]]

                    if phase == 'val':  # val data.
                        # Saving RB Plot
                        ov_image = seg.make_overlay(image, output_im)
                        name = os.path.basename(image_file)
                        image_list.append((name, ov_image))     # append ov_image name

                        name2 = name.split('.')[0] + '_green.png'

                        hard = output_im > 0.5
                        green_image = utils.fast_overlay(image, hard)
                        image_list.append((name2, green_image))  # append green_image name
                    # FN, FP, TP+FN, FP+TN
                    FN, FP, posNum, negNum = eval_image(hypes,
                                                        gt_image, output_im)

                    total_fp += FP
                    total_fn += FN
                    total_posnum += posNum
                    total_negnum += negNum
        # phase in ['train', 'val']:
        eval_dict[phase] = seg.pxEval_maximizeFMeasure(
            total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)

        if phase == 'val':
            start_time = time.time()
            for i in xrange(10):
                sess.run([softmax], feed_dict=feed_dict)
            dt = (time.time() - start_time)/10

    eval_list = []

    for phase in ['train', 'val']:
        eval_list.append(('[{}] MaxF1'.format(phase),
                          100*eval_dict[phase]['MaxF']))
        eval_list.append(('[{}] BestThresh'.format(phase),
                          100*eval_dict[phase]['BestThresh']))
        eval_list.append(('[{}] Average Precision'.format(phase),
                          100*eval_dict[phase]['AvgPrec']))
        # king 
        eval_list.append(('[{}] Pixel Accuracy'.format(phase),
                          100*eval_dict[phase]['accuracy']))
        eval_list.append(('[{}] IOU'.format(phase),
                          100*eval_dict[phase]['IOU']))
    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list
