# Copyright (c) 2017 InspiRED Robotics

# The project was based on David Sandberg's implementation of facenet

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

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
import numpy as np
from time import sleep
import math
from sklearn.externals import joblib
import time
import dlib
import imageio
import shutil

def main(args):


    # Store some git revision info in a text file in the log directory
    # facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))  
    
    #Video information
    vid = imageio.get_reader(args.input_dir,  'ffmpeg')
    vid_length = vid.get_length()
    win = dlib.image_window()
    nums=range(args.start,vid_length,args.interval)
    
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)


    for i, num in enumerate(nums):
        if i >= args.nrof_imgs:
            break
        print("Processing Frame {}".format(num))
        img = np.array(vid.get_data(num),dtype=np.uint8)
        file_name = os.path.join(output_dir,str(i)+'.jpg')
        scaled = misc.imresize(img, args.image_ratio, interp='bilinear')
        imageio.imwrite(file_name, scaled)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, help='Directory with unaligned images.')

    parser.add_argument('--output_dir', type=str, help='Directory with unaligned images.')

    parser.add_argument('--image_ratio', type=float,
        help='Resize ratio', default=0.3)
    parser.add_argument('--start', type=int,
        help='Frame start', default=10)
    parser.add_argument('--nrof_imgs', type=int,
        help='Number of total images', default=20)     
    parser.add_argument('--interval', type=int,
        help='Intervals ', default=3)  
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
