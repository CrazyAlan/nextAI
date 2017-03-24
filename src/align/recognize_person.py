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


def main(args):
    sleep(random.random())

    # Store some git revision info in a text file in the log directory
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    dataset = facenet.get_dataset(args.input_dir)
    
    print('Creating networks and loading parameters')
    # Load extracting feature model
    print('Model directory: %s' % args.model_dir)
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
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

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    
    #Array to store all image faces 
    imgs_set = []
    ## Loaidng extract feature graph

    nrof_images_total = 0
    nrof_successfully_aligned = 0
    t=time.time()

    for cls in dataset:
        for image_path in cls.image_paths:
            nrof_images_total += 1
            # filename = os.path.splitext(os.path.split(image_path)[1])[0]
            # print(image_path)
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim<2:
                    print('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:,:,0:3]

                bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces>0:
                    det = bounding_boxes[:,0:4]
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces>1:
                        bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                        img_center = img_size / 2
                        offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                        offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                        index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                        det = det[index,:]
                    det = np.squeeze(det)
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0]-args.margin/2, 0)
                    bb[1] = np.maximum(det[1]-args.margin/2, 0)
                    bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                    bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                    scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                    nrof_successfully_aligned += 1
                    # misc.imsave(output_filename, scaled)
                    imgs_set.append(scaled)

                            
    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    
    #Extracting features 
    print('\n\nStarting to extract features')
    print('Crop pics spend %.3f seconds'% (time.time()-t))
    # Run forward pass to calculate embeddings
    t=time.time()
    feed_dict = { images_placeholder:imgs_set, phase_train_placeholder:False }
    emb_array = sess.run(embeddings, feed_dict=feed_dict)
    print('Extract feature spend %.3f seconds'% (time.time()-t))

    ##Run SVM to predict images
    svm_model = joblib.load(os.path.join(os.path.expanduser(args.svm_model_dir),'model.pkl')) 
    t=time.time()    
    predicted_label = svm_model.predict(emb_array)
    print('Classifier spend %.3f seconds'% (time.time()-t))
    print('Predicted Persons')
    print(predicted_label)
    # if not os.path.isdir(args.feature_dir):  # Create the feature directory if it doesn't exist
    #     os.makedirs(args.feature_dir)
    # file_name = os.path.join(os.path.expanduser(args.feature_dir),args.feature_name)
    # np.savez(file_name, emb_array=emb_array, label=label_list)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, help='Directory with unaligned images.')

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
