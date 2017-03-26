"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
import math
from sklearn.externals import joblib
import time
import dlib
import imageio
from multiprocessing import Process, Pipe, Lock   
import threading


def main(args):


    # Store some git revision info in a text file in the log directory
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))

    detector = dlib.get_frontal_face_detector()
    
    print('Creating networks and loading parameters')
    # Load extracting feature model
    print('Model directory: %s' % args.model_dir)
    
    #Video information
    trackers = []
    positions = dlib.rectangles()
    tracker = dlib.correlation_tracker()
    vid = imageio.get_reader(args.input_video,  'ffmpeg')
    win = dlib.image_window()
    nums=range(40,vid.get_length())

    #Multi Process Info
    proc = None
    nrof_person = 0
    parent_conn, child_conn = Pipe()

    #Detection Interval
    interval = 20

    ##SVM model to predict images
    svm_model = joblib.load(os.path.join(os.path.expanduser(args.svm_model_dir),'model.pkl')) 
    person_label = []

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            
            #Loading extracting feature network
            print('Loading network used to extract features')
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
            image_size = images_placeholder.get_shape()[1]
            embedding_size = embeddings.get_shape()[1]
    
    t=time.time()

    for num in nums:
        print("Processing Frame {}".format(num))
        time.sleep(0.01)
        img = np.array(vid.get_data(num),dtype=np.uint8)
        if num%interval == 0:
            if num+interval >= nums[-1]:
                break
            img_next = np.array(vid.get_data(num+interval),dtype=np.uint8)
            
            if proc != None:
                dets = parent_conn.recv()
                proc.join()
                if len(dets) != nrof_person:
                    update_tracker(trackers, dets, img)  
                    image_sets = crop_image(img, dets, args)
                    print(np.shape(image_sets))
                    feed_dict = { images_placeholder:image_sets, phase_train_placeholder:False }
                    person_label = predicted_label(sess, feed_dict, embeddings, svm_model)

                    # t = threading.Thread(target=predicted_label, args = (sess, feed_dict, embeddings, svm_model))
                    # t.daemon = True
                    # t.start()

                nrof_person = len(dets)

            proc = Process(target=detect_resize, args=(child_conn, img_next, detector, ))
            proc.start()
            # scaled, dets = detect_resize('d', img, detector, args, CROP_IMAGE=True)
            # update_tracker(trackers, dets, img) 

            

        else:
        # Else we just attempt to track from the previous frame
            positions.clear()
            if len(trackers) > 0:
                for tracker in trackers:
                    tracker.update(img)
                    d=tracker.get_position()
                    positions.append(dlib.rectangle(int(d.left()), int(d.top()), int(d.right()), int(d.bottom())))


        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(positions, color=dlib.rgb_pixel(0,254,0))

        win.set_title('-'.join(person_label))
        # dlib.hit_enter_to_continue()
    proc.join()    

    # print('Total number of images: %d' % nrof_images_total)
    # print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    
    # #Extracting features 
    # print('\n\nStarting to extract features')
    # print('Crop pics spend %.3f seconds'% (time.time()-t))
    # # Run forward pass to calculate embeddings
    # t=time.time()
    
    # print('Extract feature spend %.3f seconds'% (time.time()-t))

    # ##Run SVM to predict images
  
    # predicted_label = svm_model.predict(emb_array)
    # print('Classifier spend %.3f seconds'% (time.time()-t))
    # print('Predicted Persons')
    # print(predicted_label)
    # if not os.path.isdir(args.feature_dir):  # Create the feature directory if it doesn't exist
    #     os.makedirs(args.feature_dir)
    # file_name = os.path.join(os.path.expanduser(args.feature_dir),args.feature_name)
    # np.savez(file_name, emb_array=emb_array, label=label_list)


def update_tracker(trackers, dets, img):
    trackers[:] = []
    for i, det in enumerate(dets):        
        trackers.append(dlib.correlation_tracker())
        trackers[i].start_track(img, det)

def predicted_label(sess, feed_dict, embeddings, svm_model):
    
    # Predicting people label
    # feed_dict = { images_placeholder:[scaled], phase_train_placeholder:False }
    t = time.time()
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    person_label = svm_model.predict(emb_array)
    print('Predicted Persons, spending time ', time.time()-t)
    print(person_label)
    return person_label

def detect_resize(conn, img, detector):
    
    if img.ndim<2:
        print('Unable to align "%s"' % image_path)
        return
    if img.ndim == 2:
        img = facenet.to_rgb(img)
    img = img[:,:,0:3]

    dets = detector(img, 1)
    conn.send(dets)


def crop_image(img, dets, args):
    bounding_boxes = []

    for i, d in enumerate(dets):
        bounding_boxes.append([d.left(), d.top(), d.right(), d.bottom()])
        
    bounding_boxes=np.asarray(bounding_boxes)

    nrof_faces = bounding_boxes.shape[0]
    
    if nrof_faces < 0:
        return

    image_sets = []
    img_size = np.asarray(img.shape)[0:2]
    bb = np.zeros(4, dtype=np.int32)

    for i in xrange(nrof_faces):

        bb[0] = np.maximum(bounding_boxes[i,0]-args.margin/2, 0)
        bb[1] = np.maximum(bounding_boxes[i,1]-args.margin/2, 0)
        bb[2] = np.minimum(bounding_boxes[i,2]+args.margin/2, img_size[1])
        bb[3] = np.minimum(bounding_boxes[i,3]+args.margin/2, img_size[0])

        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
        scaled=facenet.pre_precoss_data(scaled, False, False)
        image_sets.append(scaled)
    return image_sets

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_video', type=str, help='Path to video.', default='/cs/vml2/xca64/GitHub/pics/me.mp4')

    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
   
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--model_dir', type=str,
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters')
    parser.add_argument('--svm_model_dir', type=str, default='~/remote/models/svm',
        help='Path to save the trained model')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    print('END')
