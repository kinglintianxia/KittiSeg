#!/usr/bin/env python
# coding: utf-8

"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]

--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections  # python build-in

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf


flags = tf.app.flags # https://www.jianshu.com/p/55cbd3753ee8
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')  # 优先顺序，序号从0开始,加入的也是临时搜索路径，程序退出后失效。

from seg_utils import seg_utils as seg # submodules/evaluation/kitti_devkit/seg_utils.py

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)

# args.
flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')


default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run) # RUNS/KittiSeg_pretrained

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return
      
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    download_name = tv_utils.download(weights_url, runs_dir)
    logging.info("Extracting KittiSeg_pretrained.zip")

    import zipfile
    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def main(_):
    tv_utils.set_gpus_to_use()

    if FLAGS.input_image is None:
        logging.error("No input_image was given.")
        logging.info(
            "Usage: python demo.py --input_image data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")
        exit(1)

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run) # logdir: 'RUNS/KittiSeg_pretrained'
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    # get 'dirs' absolute path.
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir) # get a dict modules. 
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        # Run one evaluation against the full epoch of data.
        # prediction['logits']; prediction['softmax'].
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        saver = tf.train.Saver()

        # Load weights from logdir
        # Load the weights of a model stored in saver.
        core.load_weights(logdir, sess, saver) # logdir: 'RUNS/KittiSeg_pretrained'

        logging.info("Weights loaded successfully.")

    input_image = FLAGS.input_image
    logging.info("Starting inference using {} as input".format(input_image))

    # Load and resize input image.
    image = scp.misc.imread(input_image)    # RGB
    if hypes['jitter']['reseize_image']:    # Default: false
        # Resize input only, if specified in hypes
        image_height = hypes['jitter']['image_height']
        image_width = hypes['jitter']['image_width']
        image = scp.misc.imresize(image, size=(image_height, image_width),
                                  interp='cubic')

    # Run KittiSeg model on image
    feed = {image_pl: image}
    softmax = prediction['softmax']  # softmax
    output = sess.run([softmax], feed_dict=feed)    # Get probility image.

    # Reshape output from flat vector to 2D Image
    shape = image.shape
    # output[0]:image, output[1]:label
    output_image = output[0][:, 1].reshape(shape[0], shape[1])  

    # Plot confidences as red-blue overlay
    rb_image = seg.make_overlay(image, output_image)    # 0.4*output_iamge + 0.6*image

    # Accept all pixel with conf >= 0.5 as positive prediction
    # This creates a `hard` prediction result for class street
    threshold = 0.5
    street_prediction = output_image > threshold    # binary image.

    # Plot the hard prediction as green overlay
    green_image = tv_utils.fast_overlay(image, street_prediction)

    # Save output images to disk.
    if FLAGS.output_image is None:
        output_base_name = input_image
    else:
        output_base_name = FLAGS.output_image

    raw_image_name = output_base_name.split('.')[0] + '_raw.png'
    rb_image_name = output_base_name.split('.')[0] + '_rb.png'
    green_image_name = output_base_name.split('.')[0] + '_green.png'

    scp.misc.imsave(raw_image_name, output_image)
    scp.misc.imsave(rb_image_name, rb_image)
    scp.misc.imsave(green_image_name, green_image)

    logging.info("")
    logging.info("Raw output image has been saved to: {}".format(
        os.path.realpath(raw_image_name)))
    logging.info("Red-Blue overlay of confs have been saved to: {}".format(
        os.path.realpath(rb_image_name)))
    logging.info("Green plot of predictions have been saved to: {}".format(
        os.path.realpath(green_image_name)))

    logging.info("")
    logging.warning("Do NOT use this Code to evaluate multiple images.")

    logging.warning("Demo.py is **very slow** and designed "
                    "to be a tutorial to show how the KittiSeg works.")
    logging.warning("")
    logging.warning("Please see this comment, if you like to apply demo.py to"
                    "multiple images see:")
    logging.warning("https://github.com/MarvinTeichmann/KittiBox/"
                    "issues/15#issuecomment-301800058")


if __name__ == '__main__':
    tf.app.run()    # 上述第一行代码表示如果当前是从其它模块调用的该模块程序，则不会运行main函数！而如果就是直接运行的该模块程序，则会运行main函数。
