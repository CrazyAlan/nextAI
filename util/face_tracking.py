
import os
import glob
import numpy as np
import dlib
from scipy import misc
import imageio
import time
import threading
import copy
from multiprocessing import Process, Pipe, Lock   


def detect_face(conn, detector, img):
    
    # acquired = lock.acquire(True)
    # print('Start process')   
    dets = detector(img, 1)
    conn.send(dets)

def update_tracker(trackers, dets, img):
    trackers[:] = []
    for i, det in enumerate(dets):        
        trackers.append(dlib.correlation_tracker())
        trackers[i].start_track(img, det)


detector = dlib.get_frontal_face_detector()

# Path to the video frames
filename = '/cs/vml2/xca64/GitHub/pics/Two.mov'

# Create the correlation tracker - the object needs to be initialized
# before it can be used
vid = imageio.get_reader(filename,  'ffmpeg')

win = dlib.image_window()
nums=range(100,280,1)
# We will track the frames as we load them off of disk
trackers = []
positions = dlib.rectangles()

trackerLock = Lock()

image1 = np.array(vid.get_data(10),dtype=np.uint8)
parent_conn, child_conn = Pipe()

proc = None
no_of_person = 0

for num in nums:
    print("Processing Frame {}".format(num))
    img = np.array(vid.get_data(num),dtype=np.uint8)
    
    time.sleep(0.02)

    # We need to initialize the tracker on the first frame
    if num%10 == 0:
        # Start a track on the juice box. If you look at the first frame you
        # will see that the juice box is contained within the bounding
        # box (74, 67, 112, 153).
        img_next_10 = np.array(vid.get_data(num+10),dtype=np.uint8)

        if proc != None:
            dets = parent_conn.recv()
            proc.join()
            if len(dets) != no_of_person:
                update_tracker(trackers, dets, img)    
            no_of_person = len(dets)

        proc = Process(target=detect_face, args=(child_conn, detector, img_next_10 ))
        proc.start()
        
        #Doesn't update if no number changes
        
        # img2 = np.random.randint(255, size=(1000,1000,3))
        # t = threading.Thread(target=updatae_tracker, args = (trackerLock, detector, img, trackers))
        # t.daemon=True
        # t.start()
        
    else:
        # Else we just attempt to track from the previous frame
        positions.clear()
        # trackerLock.acquire()
        # print('Is the process alive', proc.is_alive())          

        if len(trackers) > 0:
            for tracker in trackers:
                tracker.update(img)
                d=tracker.get_position()
                positions.append(dlib.rectangle(int(d.left()), int(d.top()), int(d.right()), int(d.bottom())))
        # trackerLock.release()

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(positions)
    # dlib.hit_enter_to_continue()


proc.join()

if __name__ == '__main__':
    main()