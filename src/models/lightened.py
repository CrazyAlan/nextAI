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

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import models.network as network

import tensorflow.contrib.slim as slim

def inference(images, keep_probability, phase_train=True, weight_decay=0.0):
    """ Define an inference network for face recognition based 
           on inception modules using batch normalization
    
    Args:
      images: The images to run inference on, dimensions batch_size x height x width x channels
      phase_train: True if batch normalization should operate in training mode
    """
    endpoints = {}

    net = network.convMfm(images, 3, 48, 9, 9, 1, 1, 'VALID', 'conv1_9x9', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv1_9x9'] = net
    net = network.mpool(net, 2, 2, 2, 2, 'SAME', 'pool1')
    endpoints['pool1'] = net

    net = network.convMfm(images, 48, 96, 5, 5, 1, 1, 'VALID', 'conv2_5x5', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv2_5x5'] = net
    net = network.mpool(net, 2, 2, 2, 2, 'SAME', 'pool2')
    endpoints['pool2'] = net

    net = network.convMfm(images, 96, 128, 5, 5, 1, 1, 'VALID', 'conv3_5x5', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv3_5x5'] = net
    net = network.mpool(net, 2, 2, 2, 2, 'SAME', 'pool3')
    endpoints['pool3'] = net

    net = network.convMfm(images, 128, 192, 4, 4, 1, 1, 'VALID', 'conv4_4x4', phase_train=phase_train, use_batch_norm=True, weight_decay=weight_decay)
    endpoints['conv4_4x4'] = net
    net = network.mpool(net, 2, 2, 2, 2, 'SAME', 'pool4')
    endpoints['pool4'] = net

    with tf.variable_scope('Logits'):
        net = slim.flatten(net)
        net = slim.dropout(net, keep_probability, is_training=is_training,
                            scope='Dropout')   
        end_points['PreLogitsFlatten'] = net
         
    return net, endpoints
