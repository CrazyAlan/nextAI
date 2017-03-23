from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import os
import numpy as np
import operator
import shutil
import sys


def main(args):

    persons = os.listdir(os.path.expanduser(args.data_dir))
    name_dic = {}
    for person in persons:
        person_folder=os.path.join(os.path.expanduser(args.data_dir),person)
        if not os.path.isdir(person_folder):
            print('not folder')
            continue
        # person_folder=os.path.join(os.path.expanduser(args.data_dir),person)
        name_dic[person] = len(os.listdir(person_folder))

    sorted_x = sorted(name_dic.items(), key=operator.itemgetter(1))

    if not os.path.isdir(args.output_dir):  # Create the output directory if it doesn't exist
        os.makedirs(args.output_dir)

    for i in range(1,args.size+1):

        org=os.path.join(os.path.expanduser(args.data_dir),sorted_x[-i][0])
        dst=os.path.join(os.path.expanduser(args.output_dir),sorted_x[-i][0])

        if not os.path.isdir(dst):  # Create the output directory if it doesn't exist
            os.makedirs(dst)

        img_list = os.listdir(org)
        for j in range(args.per_imgs):
            org_file = os.path.join(os.path.expanduser(org),img_list[j])
            dst_file = os.path.join(os.path.expanduser(dst),img_list[j])
            shutil.copyfile(org_file, dst_file)

        print(dst)
        # print sorted_x

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='~/remote/datasets',
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--output_dir', type=str, default='~/remote/datasets',
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--size', type=int,
        help='Number of images to copy to store', default=10)
    parser.add_argument('--per_imgs', type=int,
        help='Number of images to for each pearson', default=20)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))