'''
Item-item similarity based collaborative filtering model
References: Factor in the Neighbors: Scalable and Accurate Collaborative Filtering (Koren 2008)
            Item-Based Collaborative Filtering Recommendation Algorithms (Sarwar, Karypis, Konstan, Riedl 2010)

Will try to learn global weights for explicit and implicit similarity measures using gradient descent.

Expect options...
-p    The prefix used to get the ratings and validation data as well as save new data
-d    The distance type to use. Can take 'pearson' or 'cosine'
-k    The neighborhood size to use.
-l    The lambda value used to penalize similaities made with few users

Example call:
ipython collaborative_filtering_similarity.py -k -1 -d pearson --prefix data_20u_1b -l 50
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
parser.add_option("-d", "--distance", action="store", 
                  dest="distance", type="string", default=None, 
                  help="type of distance to use")
parser.add_option("-k", action="store", 
                  dest="k", type="string", default=None, 
                  help="neighbor hood size")
parser.add_option("-l", "--lambda", action="store", 
                  dest="l4", type="string", default=None, 
                  help="lambda parameter for penalizing weights made with few users")

def main():
    # Get command line options
    options, args = parser.parse_args()
    prefix = options.prefix
    neighbor_hood_size = int(options.k)
    distance = options.distance

    # Get the ratings matrix
    R = np.array(pickle.load(open(pickle_directory + prefix + '_stars_train.p')).todense())

    # Dimensions
    m = R.shape[0]      # number of users
    n = R.shape[1]      # number of businesses
    if neighbor_hood_size == -1:
        neighbor_hood_size = n

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
    
    # Center the data
    if distance == 'pearson':
        means = np.ma.average(R, axis=0, weights=R!=0).data
        non_zeros = (R != 0).astype(int)
        centerer = non_zeros * means[np.newaxis, :]
    elif distance == 'cosine':
        means = np.ma.average(R, axis=1, weights=R!=0).data
        non_zeros = (R != 0).astype(int)
        centerer = non_zeros * means[:, np.newaxis]
    R = R - centerer
    
    # Initialize the Pearson Corr. Coef. Matrix
    P = np.zeros(n * n).reshape(n, n)

    # Parameters to tweak the corr. coeff.
    l4 = int(options.l4)

    # For each item, get it's similarity to other businesses
    print '\nBuilding similarity matrix ... '
    try:
        P = pickle.load(open(pickle_directory + prefix + '_P_' + distance + '_k' + str(neighbor_hood_size) + '_l' + str(l4) + '.p'))
        P = np.array(P.todense())
    except:
        for i in range(n):
            # Get all the users that rated this business
            users_i = business_buckets[i]
            
            # Go through all other business, but only fill in the lower triangle
            # This will result in (N choose 2) instead of N^2 run time
            for j in range(i+1, n):
                # Prepare the fraction pieces
                numerator = 0
                denominator_left = 0
                denominator_right = 0

                # Get the users in common
                us = np.intersect1d(users_i, business_buckets[j])

                # Iterate through these users and do the sums for the similarity measure
                if len(us) != 0:
                    eta = 0
                    for u in us:
                        l = R[u, i]
                        r = R[u, j]

                        numerator += l * r
                        denominator_left += np.power(l, 2)
                        denominator_right += np.power(r, 2)
                        denominator = np.sqrt(denominator_left) * np.sqrt(denominator_right)

                        eta += 1
                    
                    # If the denominator is zero, then the ratings minus mean turned out to be zero for both items
                    if denominator != 0: 
                        P[i,j] = float(eta) / (eta + l4) * float(numerator) / denominator
                        #P[i,j] = float(numerator) / denominator
                    else:
                        P[i,j] = 0

            # Print the progress
            if i % 10 == 0 or i == (n - 1):
                sys.stdout.write('\r' + str(round(float(i)/(n - 1) * 100, 2)) + '%    ')
                sys.stdout.flush()
    print 'Done'

    print '\nCleaning similarity matrix ... '
    # Clean up the corr. matrix
    P[P < 0] = 0
    P = P + P.T

    # Only keep k-most similar
    for i in range(n):
        modifier = np.zeros(n)
        modifier[np.argsort(P[i,:])[::-1][:neighbor_hood_size]] = 1
        P[i, :] = P[i, :] * modifier

    # Store P
    P = scipy.sparse.lil_matrix(P)
    pickle.dump(P, open(pickle_directory + prefix + '_P_' + distance + '_k' + str(neighbor_hood_size) + '_l' + str(l4) + '.p', "wb"))
    print 'Stored similarity matrix into ' + pickle_directory + prefix + '_P_' + distance + '_k' + str(neighbor_hood_size) + '_l' + str(l4) + '.p'
    P = np.array(P.todense())
    print 'Done'

    # Uncenter the ratings matrix
    R = R + centerer

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
        
        # Initialize the fraction
        num = 0
        denom = 0

        # Get all the items in common with this item
        for j in np.intersect1d(np.where(P[i,:] != 0)[0], user_buckets[u]):
            num += P[i,j] * (R[u,j] - (mu + b_i[j] + b_u[u]))
            denom += P[i,j]

        # If we can't predict, fill it with the mean
        try:
            r = (mu + b_i[i] + b_u[u]) + num/denom
            nz += 1
        except:
            r = mu + b_i[i] + b_u[u]

        tot += 1

        # Insert the prediction
        R[u,i] = r

        if k % 10 == 0 or k == (len(required_row) - 1):
            sys.stdout.write('\r' + str(round(float(k)/(len(required_row) - 1) * 100, 2)) + '%    ')
            sys.stdout.flush()
    print 'Done'

    print "\nNumber of non-zero predictions made: " + str(float(nz)/tot)

    # Get the RMSE
    R = scipy.sparse.csr_matrix(R)
    print '\nRMSE for CF (prefix = ' + prefix + ', k = ' + str(neighbor_hood_size) + '): \t' + str(tools.rmse(R, V))

if __name__ == '__main__':
    main()
