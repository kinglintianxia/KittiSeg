#! /usr/bin/env python
# coding: utf-8
# @Author: king@2018.10
"""
Detects roads in an image using KittiSeg.

Usage:
# for live camera.
python run_live.py 
# for read images.
python run_live.py --image_dir your-dir

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# for tx2 camera on board.
import cv2
import ensemble as ensemble
import numpy as np

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf
import time
import PIL.Image as Image


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

from seg_utils import seg_utils as seg

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


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('image_dir', None,
                    'Image to apply KittiSeg.')
default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

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

# CORE function.
def create_test_output(sess, image_pl, image, prediction, image_file_suf):
	shape = image.shape
	feed = {image_pl:image}

	softmax = prediction['softmax']
	output = sess.run([softmax], feed_dict=feed)

	# Reshape output from flat vector to 2D Image
	shape = image.shape
	output_image = output[0][:, 1].reshape(shape[0], shape[1])
	# cv2.imshow("bin_img", output_image)
	#print ("output_image.type, output_image.dtype, output_image.min, output_image.max", type(output_image), output_image.dtype, np.min(output_image), np.max(output_image))
	#scp.misc.imsave('./bin_img.png', output_image)

	# Plot confidences as red-blue overlay
	rb_image = seg.make_overlay(image, output_image)
	# cv2.imshow("rb_img", rb_image)
	# Accept all pixel with conf >= 0.5 as positive prediction
	# This creates a `hard` prediction result for class street
	threshold = 0.5
	street_prediction = output_image > threshold

	# Plot the hard prediction as green overlay
	green_image = tv_utils.fast_overlay(image, street_prediction)
	# cv2.imshow("green_img", green_image)

	#########################
	# hough lines.
	# bin_img = np.expand_dims(output_image, axis=3)
	# color_img = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR, 3);
	# color_img = np.stack((output_image,) * 3, axis=-1)
	# color_img = color_img.astype(np.uint8)
	# print ("---output_image.shape---", bin_img.shape)
	#bin_img = cv2.imread('./bin_img.png')
	###########!!!!!!!!!!!!######################
	bin_img = (output_image*255).astype(np.uint8)
	bin_img = np.stack((bin_img,)*3, axis=-1)
	###########!!!!!!!!!!!!######################

	line_img = ensemble.process_an_image(bin_img, image, 0)
	# height > width

	if float(shape[1]/shape[0]) >= 1.5:	
		green_line_img = np.vstack((green_image, line_img))
	else:
		green_line_img = np.hstack((green_image, line_img))
	# putting text on image.
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	topLeftCornerOfText = (30,30)
	fontScale              = 1
	fontColor              = (255,255,255)
	lineType               = 2

	cv2.putText(green_line_img,
		image_file_suf, 
		topLeftCornerOfText, 
		font, 
		fontScale,
		fontColor,
		lineType)
	# cv2.imshow('line_img', green_line_img)

	return green_line_img
	# Save output images to disk.
	logging.info("Inference {} Done.".format(image_file_suf))
	
	

def main(_):
	tv_utils.set_gpus_to_use()
	if FLAGS.image_dir is not None:
		logging.info("---inference with image files.---")
	else:
		# open tx2 camera.
		print ("---Open Camera...---")
		cap=cv2.VideoCapture(0) 
		if not cap.isOpened():
			sys.exit("Failed to open camera!")

	if FLAGS.logdir is None:
		# Download and use weights from the MultiNet Paper
		if 'TV_DIR_RUNS' in os.environ:
		    runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
		                            'KittiSeg')
		else:
		    runs_dir = 'RUNS'
		maybe_download_and_extract(runs_dir)
		logdir = os.path.join(runs_dir, default_run)
	else:
		logging.info("Using weights found in {}".format(FLAGS.logdir))
		logdir = FLAGS.logdir

	# Loading hyperparameters from logdir
	hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

	logging.info("Hypes loaded successfully.")

	# Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
	modules = tv_utils.load_modules_from_logdir(logdir)
	logging.info("Modules loaded successfully. Starting to build tf graph.")

	# Create tf graph and build module.
	with tf.Graph().as_default():
		# Create placeholder for input
		image_pl = tf.placeholder(tf.float32)
		# add a batch dimension.
		image = tf.expand_dims(image_pl, 0)	# [0, 480, 640, 3]

		# build Tensorflow graph using the model from logdir
		prediction = core.build_inference_graph(hypes, modules,
		                                        image=image)

		logging.info("Graph build successfully.")

		# Create a session for running Ops on the Graph.
		sess = tf.Session()
		saver = tf.train.Saver()

		# Load weights from logdir
		core.load_weights(logdir, sess, saver)

		logging.info("Weights loaded successfully.")
		
		# Read images from disk.
		if FLAGS.image_dir is not None:
			image_dir = FLAGS.image_dir
			
			#1 Given testing.txt, ie. '/home/nvidia/Documents/runwaydetection/data_road/testing.txt'
			if image_dir.find('.txt') != -1:	# 'data_road/testing.txt'
				image_dir_pre = image_dir.split('testing')[0] # /home/nvidia/Documents/runwaydetection/data_road/
				image_file_suf = image_dir.split('testing')[1]	# '.txt'
				print ("image_file_suf: ", image_file_suf)
				with open(image_dir) as file:	# 'data_road/testing.txt'
					for i, image_file in enumerate(file):	# image_file="testing/image_2/1_000001.png"
							image_file_suf = image_file.rstrip()	# delete space ' '.
							image_file = os.path.join(image_dir_pre, image_file_suf)	# './data_road/testing/image_2/1_000001.png'
							#image = Image.open(image_file)
							
							image = scp.misc.imread(image_file)							
							green_img = create_test_output(sess, image_pl, image, prediction, image_file_suf)
							cv2.imshow("road_image", green_img)
							
							# press 'q' to quite.
							key = cv2.waitKey(0)
							if (key&0xff == ord('s')):
								img_name = image_file_suf.split('image_2')[1]	# /1_000001.png
								img_name = "../seg_img"+img_name;	# '../seg_img/1_000001.png'
								scp.misc.imsave(img_name, green_img)
							elif (key&0xff == ord('q')):
								break;
							# if cv2.waitKey(1) & 0xff == ord('q'):
							# 	break; 

			#2 Given a directory. ie. '/home/nvidia/Documents/runwaydetection/data_road/testing/image_2'		
			else:
				os.chdir(image_dir)		# change directory to 'data_road/testing/image_2'
				path = os.getcwd()		# get current work directory.
				files = os.listdir(path)	# 1.png
				files.sort()	# sort in order
				#print ("files: ", files)
				for afile in files:	# *.png
					file_path=os.path.join(path,afile)	# '/home/nvidia/Documents/runwaydetection/data_road/testing/image_2/*.png'
					if os.path.isfile(file_path):
						if os.path.getsize(file_path)==0:
							continue
						image = scp.misc.imread(file_path)
						green_img = create_test_output(sess, image_pl, image, prediction, afile)
						cv2.imshow("road_image", green_img)
						# press 'q' to quite.
						
						if cv2.waitKey(1) & 0xff == ord('q'):
							break; 

						
						#cv2.destroyAllWindows()
		# run live with tx2 camera.		        
		else: 	# run live
			#os.system("python tx2_cam_save.py")
			cnt = 0
			while (1):
				cnt += 1
		###################################################
				# Load and resize input image
				ret, image = cap.read() # show a frame   
				if not ret:
					break;
				# cv2.imshow("capture", frame) 
				#cv2.imwrite("./tx2_img.png", frame)
				#image = scp.misc.imread("./tx2_img.png")
				if image is None:
					break;
				# 因为opencv读取进来的是BGR顺序呢的, TF:rgb
				image = image[..., -1::-1]    # 
				green_img = create_test_output(sess, image_pl, image, prediction, str(cnt))
				cv2.imshow("road_image", green_img)
				# press 'q' to quite.0
				if cv2.waitKey(0) & 0xff == ord('q'):
					break; 

			cap.release() 
			cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()
