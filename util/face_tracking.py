

#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example shows how to use the correlation_tracker from the dlib Python
# library.  This object lets you track the position of an object as it moves
# from frame to frame in a video sequence.  To use it, you give the
# correlation_tracker the bounding box of the object you want to track in the
# current video frame.  Then it will identify the location of the object in
# subsequent frames.
#
# In this particular example, we are going to run on the
# video sequence that comes with dlib, which can be found in the
# examples/video_frames folder.  This video shows a juice box sitting on a table
# and someone is waving the camera around.  The task is to track the position of
# the juice box as the camera moves around.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import os
import glob
import numpy as np
import dlib
from scipy import misc
import imageio
import time

detector = dlib.get_frontal_face_detector()

# Path to the video frames
filename = '/cs/vml2/xca64/GitHub/pics/Two.mov'

# Create the correlation tracker - the object needs to be initialized
# before it can be used
vid = imageio.get_reader(filename,  'ffmpeg')

win = dlib.image_window()
nums=range(189,300)
# We will track the frames as we load them off of disk
trackers = []
positions = dlib.rectangles()

for num in nums:
    print("Processing Frame {}".format(num))
    img = np.array(vid.get_data(num),dtype=np.uint8)
 

    # We need to initialize the tracker on the first frame
    if num == nums[0]:
        # Start a track on the juice box. If you look at the first frame you
        # will see that the juice box is contained within the bounding
        # box (74, 67, 112, 153).
        dets = detector(img, 1)
        print(type(dets))
        trackers = []
        for i, det in enumerate(dets):        
            trackers.append(dlib.correlation_tracker())
            trackers[i].start_track(img, det)

        print('Number of dets', len(dets))
        continue
        
    else:
        # Else we just attempt to track from the previous frame
        positions.clear()
        for tracker in trackers:
            tracker.update(img)
            d=tracker.get_position()
            print d.left(), d.top(), d.right(), d.bottom()
            positions.append(dlib.rectangle(int(d.left()), int(d.top()), int(d.right()), int(d.bottom())))

    print(positions)

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(positions)
    # dlib.hit_enter_to_continue()