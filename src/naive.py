import sys
from optparse import OptionParser
import json
import cPickle as pickle
import numpy as np
import scipy.sparse
import pandas as pd
import re
import os
import pylab as plt
import tools

# pickle directory
pickle_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/pickle/')
# data directory
data_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/data/')

usage = "<script> [args ...]"

description = "Script for building data matrix objects."

parser = OptionParser(usage=usage, description=description)
parser.add_option("-m", "--method", action="store", 
                  dest="method", type="string", default=None,
                  help="method used to replace missing ratings", metavar="FILE")
parser.add_option("-p", "--prefix", action="store", 
                  dest="prefix", type="string", default=None, 
                  help="prefix for the data files", metavar="FILE")

def main():
    options, args = parser.parse_args()

    method = options.method
    prefix = options.prefix

    data_stars_train, data_stars_validate, data_stars_test = tools.import_pickles(prefix, 'stars')
    data_stars_train = np.array(data_stars_train.todense())

    if method == 'business':
        means = np.ma.average(data_stars_train, axis=0, weights=data_stars_train!=0).data
        for i in range(len(means)):
            data_stars_train[data_stars_train[:,i] == 0,i] = means[i]
    
    elif method == 'user':
        means = np.ma.average(data_stars_train, axis=1, weights=data_stars_train!=0).data
        for i in range(len(means)):
            data_stars_train[i, data_stars_train[i,:] == 0] = means[i]
    
    elif method == 'random':
        for i in range(data_stars_train.shape[0]):
            for j in range(data_stars_train.shape[1]):
                if data_stars_train[i,j] == 0:
                    data_stars_train[i, j] = np.random.random_integers(1,5)

    elif method == 'none':
        data_stars_train = data_stars_train

    else:
        print 'Unknown method.'

    data_stars_train = scipy.sparse.csr_matrix(data_stars_train)

    print "RMSE for '" + method + "': " + str(tools.rmse(data_stars_train, data_stars_validate))


if __name__ == "__main__":
    main()