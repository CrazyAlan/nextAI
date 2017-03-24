from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import os
import numpy as np
from sklearn import svm
import sys
from sklearn.externals import joblib

def train(args):
    npfile = np.load(args.npz_file_dir)
    X=npfile['emb_array']
    y=npfile['label']
    nr_of_samples=len(y)
    model=svm.SVC(kernel='rbf', C=args.C, gamma=args.gamma)

    if args.test_acc:
        for i in xrange(1,25):        
        # Divide the trainning and testing set to 9:1
            indexs=np.random.permutation(nr_of_samples)
            X_permuted=np.array([X[i,:] for i in indexs])
            y_permuted=np.array([y[i] for i in indexs])

            cut_point = int(0.9*nr_of_samples)
            X_trained=X_permuted[0:cut_point,:]
            X_test=X_permuted[cut_point:,:]
            y_trained=y_permuted[0:cut_point]
            y_test=y_permuted[cut_point:]
                

            model.fit(X_trained, y_trained)
            print('accuracy', np.count_nonzero(model.predict(X_test)==y_test)/float(len(y_test)))
    else:
        #print storing the svm model 
        print('Trainning and Storing SVM model')
        model.fit(X, y)
        print('accuracy', np.count_nonzero(model.predict(X)==y)/float(len(y)))

        if not os.path.exists(os.path.expanduser(args.svm_model_dir)):
            os.makedirs(os.path.expanduser(args.svm_model_dir))

        joblib.dump(model, os.path.join(os.path.expanduser(args.svm_model_dir),'model.pkl')) 
        
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--npz_file_dir', type=str, default='~/remote/datasets',
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--svm_model_dir', type=str, default='~/remote/datasets',
        help='Path to save the trained model')
    parser.add_argument('--test_acc', action='store_true',
        help='Just used to test svm acc')
    parser.add_argument('--C', type=float,
        help='C value for svm', default=1)
    parser.add_argument('--gamma', type=float,
        help='C value for svm', default=1)

    return parser.parse_args(argv)

if __name__ == '__main__':
    train(parse_arguments(sys.argv[1:]))