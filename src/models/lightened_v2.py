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
import tensorflow.contrib.slim as slim

def conv(inpOp, nOut, nIn, stride=2, padding='VALID', scope='conv'):
    with tf.variable_scope(scope):
        net = slim.conv2d(inpOp, nOut, nIn, stride=stride, padding=padding, scope=scope)
        net = tf.nn.elu(net)

    return net

def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return lightened_v2(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, reuse=reuse)


def lightened_v2(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='LightenedV1'):
    """Creates the Lightened V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
  
    with tf.variable_scope(scope, 'LightenedV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                endpoints = {}
                
                net=conv(inputs, 48, 9, stride=1, padding='VALID', scope='conv1_9x9')
                end_points['conv1_9x9'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='pool1')
                end_points['pool1'] = net
                
                net=conv(net, 96, 5, stride=1, padding='VALID', scope='conv2_5x5')
                end_points['conv2_5x5'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='pool2')
                end_points['pool2'] = net

                net=conv(net, 128, 5, stride=1, padding='VALID', scope='conv3_5x5')
                end_points['conv3_5x5'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='pool3')
                end_points['pool3'] = net

                net=conv(net, 192, 4, stride=1, padding='VALID', scope='conv4_4x4')
                end_points['conv4_5x5'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='pool4')
                end_points['pool4'] = net

                with tf.variable_scope('Logits'):
                        net = slim.flatten(net)
                        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                            scope='Dropout')   
                        endpoints['PreLogitsFlatten'] = net
                
    return net, end_points


