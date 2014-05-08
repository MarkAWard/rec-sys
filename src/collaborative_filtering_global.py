'''
Global collaborative filtering model
References: Factor in the Neighbors: Scalable and Accurate Collaborative Filtering (Koren 2008)

Will try to learn global weights for explicit and implicit similarity measures using gradient descent.

Expect many options...
-p    The prefix used when building the original P from the similarity collaborative filtering, also
      needed to get the ratings and validation data.
-d    The distance type used to create the original P. Can take 'pearson' or 'cosine'
-k    The neighborhood size to use.
-L    The lambda value used originally to penalize similaities made with few users

The above four options are used to properly call on and save new pickles.

You also need...
-l    The lambda value for gradient descent (step size)
-g    The gamma for the gradient descent

Example call:
ipython collaborative_filtering_global.py --prefix data_20u_1b -d pearson -L 50 -k 7279 -l 0.002 -g 0.005
'''

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
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD

# pickle directory
pickle_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/pickle/')
# data directory
data_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/data/')

usage = "<script> [args ...]"

description = "Script for building data matrix objects."

parser = OptionParser(usage=usage, description=description)
parser.add_option("-p", "--prefix", action="store", 
                  dest="prefix", type="string", default=None, 
                  help="prefix for the data files", metavar="FILE")
parser.add_option("-l", "--lambda", action="store", 
                  dest="l5", type="string", default=None, 
                  help="lambda parameter for step size")
parser.add_option("-g", "--gamma", action="store", 
                  dest="g", type="string", default=None, 
                  help="gamma parameter")
parser.add_option("-d", "--distance", action="store", 
                  dest="distance", type="string", default=None, 
                  help="type of distance to use")
parser.add_option("-k", action="store", 
                  dest="k", type="string", default=None, 
                  help="neighbor hood size")
parser.add_option("-L", "--lambda4", action="store", 
                  dest="l4", type="string", default=None, 
                  help="lambda parameter for penalizing weights made with few users")

def main():
    # Get command line options
    options, args = parser.parse_args()
    prefix = options.prefix
    distance = options.distance
    neighbor_hood_size = int(options.k)
    l4 = int(options.l4)

    # Get the ratings matrix
    R = np.array(pickle.load(open(pickle_directory + prefix + '_stars_train.p')).todense())
    
    # Dimensions
    m = R.shape[0]      # number of users
    n = R.shape[1]      # number of businesses

    # Find all the businesses rated by each user ...
    print 'Counting and binning users ... ',
    sys.stdout.flush()
    user_buckets = {}
    for i in range(0,R.shape[0]):
        businesses = np.where(R[i,:] > 0)[0]
        user_buckets[i] = businesses
    print 'Done'

    # ... and all the users that rated each business
    print 'Counting and binning businesses ... ',
    sys.stdout.flush()
    business_buckets = {}
    for i in range(0,R.shape[1]):
        users = np.where(R[:,i] > 0)[0]
        business_buckets[i] = users
    print 'Done'

    # Get the average of each column and row
    business_means = np.ma.average(R, axis=0, weights=R!=0).data
    user_means = np.ma.average(R, axis=1, weights=R!=0).data

    # Get the component-pieces of each score: overall average, item deviation, user deviation
    mu = np.ma.average(R, weights=R!=0)

    # Initialize the deviation arrays
    b_i = np.zeros(n)
    b_u = np.zeros(m)

    # Parameters for the deviations
    l2 = 25
    l3 = 10

    # Get the item deviations
    for i in range(n):
        numerator = 0
        for u in business_buckets[i]:
            numerator += (R[u, i] - mu)
        b_i[i] = float(numerator) / (l2 + len(business_buckets[i]))

    # Get the user deviations
    for u in range(m):
        numerator = 0
        for i in user_buckets[u]:
            numerator += (R[u, i] - mu - b_i[i])
        b_u[u] = float(numerator) / (l3 + len(user_buckets[u]))


    # Get all the matrices we'll need
    R = pickle.load(open(pickle_directory + prefix + '_stars_train.p'))
    P = pickle.load(open(pickle_directory + prefix + '_P_' + distance + '_k' + str(n) + '_l' + str(l4) + '.p'))
    P = np.array(P.todense())
    C = np.array(P, copy = True)

    R_row = R.nonzero()[0]
    R_col = R.nonzero()[1]

    # Parameters for the global model
    g = float(options.g)
    l5 = float(options.l5)

    # Iterate and update the weights for explicit and implicit
    for iteration in range(20):
        print 'Iteration ' + str(iteration + 1)
        for k, u in enumerate(R_row):
            i = R_col[k]
            
            sigma_r = 0
            sigma_n = 0

            intersection = np.intersect1d(np.where(P[i,:] != 0)[0], user_buckets[u])
            
            for j in intersection:
                sigma_r += (R[u, j] - (mu + b_i[j] + b_u[u])) * P[i, j]
                sigma_n += C[i, j]

            if len(intersection) == 0:
                pow1 = 0
            else:
                pow1 = np.power(len(intersection), -0.5)

            r_hat = mu + b_u[u] + b_i[i] + pow1 * sigma_r + pow1 * sigma_n
            
            err = R[u, i] - r_hat

            b_u[u] += g * (err - l5 * b_u[u])
            b_i[i] += g * (err - l5 * b_i[i])

            for j in intersection:
                P[i, j] += g * (np.power(len(intersection), -0.5) * err * (R[u, j] - (mu + b_u[u] + b_i[j])) - l5 * P[i, j])
                C[i, j] += g * (np.power(len(intersection), -0.5) * err - l5 * C[i, j])
            if k % 10 == 0 or k == (len(R_row) - 1):
                sys.stdout.write('\r' + str(round(float(k)/(len(R_row) - 1) * 100, 2)) + '%    ')
                sys.stdout.flush()
        print 'Done\n'

    # Clean the weights
    P[P < 0] = 0
    C[C < 0] = 0

    # Only keep k-most similar
    for i in range(n):
        modifier = np.zeros(n)
        modifier[np.argsort(P[i,:])[::-1][:neighbor_hood_size]] = 1
        P[i, :] = P[i, :] * modifier
        C[i, :] = C[i, :] * modifier

    # Store similarities
    P = scipy.sparse.lil_matrix(P)
    C = scipy.sparse.lil_matrix(C)
    pickle.dump(P, open(pickle_directory + prefix + '_W_k' + str(neighbor_hood_size) + '_l' + str(l5) + '_g' + str(g) + '.p', "wb"))
    pickle.dump(C, open(pickle_directory + prefix + '_C_k' + str(neighbor_hood_size) + '_l' + str(l5) + '_g' + str(g) + '.p', "wb"))
    print 'Stored W matrix into ' + pickle_directory + prefix + '_W_k' + str(neighbor_hood_size) + '_l' + str(l5) + '_g' + str(g) + '.p'
    print 'Stored C matrix into ' + pickle_directory + prefix + '_C_k' + str(neighbor_hood_size) + '_l' + str(l5) + '_g' + str(g) + '.p'
    P = np.array(P.todense())
    C = np.array(C.todense())
    print 'Done'

    # Get the validation matrix
    V = pickle.load(open(pickle_directory + prefix + '_stars_validate.p'))

    # Which cells do we need to predict?
    required_row = V.nonzero()[0]
    required_col = V.nonzero()[1]

    # Keep track of our predictions
    nz = 0
    tot = 0

    # Start predicting for all the required cells
    print '\nMaking new predictions ... '
    for k, u in enumerate(required_row):
        i = required_col[k]
        
        intersection = np.intersect1d(np.where(P[i,:] != 0)[0], user_buckets[u])
        
        for j in intersection:
            sigma_r += (R[u, j] - (mu + b_i[j] + b_u[u])) * P[i, j]
            sigma_n += C[i, j]

        if len(intersection) == 0:
            pow1 = 0
        else:
            pow1 = np.power(len(intersection), -0.5)

        r_hat = mu + b_u[u] + b_i[i] + pow1 * sigma_r + pow1 * sigma_n


        # Insert the prediction
        R[u,i] = r_hat

        if k % 10 == 0 or k == (len(required_row) - 1):
            sys.stdout.write('\r' + str(round(float(k)/(len(required_row) - 1) * 100, 2)) + '%    ')
            sys.stdout.flush()
    print 'Done'

    # Get the RMSE
    R = scipy.sparse.csr_matrix(R)
    print '\nRMSE for global (prefix = ' + prefix + ', k = ' + str(neighbor_hood_size) + ', l = ' + str(l5) + ', g = ' + str(g) + '): \t' + str(tools.rmse(R, V))

if __name__ == '__main__':
    main()
