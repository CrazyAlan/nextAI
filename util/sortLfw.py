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
		name_dic[person] = len(os.listdir(person_folder))

	sorted_x = sorted(name_dic.items(), key=operator.itemgetter(1))

	if not os.path.isdir(args.output_dir):  # Create the output directory if it doesn't exist
                os.makedirs(args.output_dir)

	for i in range(1,args.size+1):

		org=os.path.join(os.path.expanduser(args.data_dir),sorted_x[-i][0])
		dst=os.path.join(os.path.expanduser(args.output_dir),sorted_x[-i][0])
		shutil.copytree(org, dst)
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
	return parser.parse_args(argv)

if __name__ == '__main__':
	main(parse_arguments(sys.argv[1:]))