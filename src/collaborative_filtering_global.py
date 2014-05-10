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
parser.add_option("-t", "--training", action="store_false", 
                  dest="train", default=True, 
                  help="should training take place, or should we just grab the pretrained weights")
parser.add_option("-c", "--continue", action="store_true", 
                  dest="cont", default=False, 
                  help="continue training with weight matrices that already exist?")
parser.add_option("-f", "--fake_implicit", action="store_true", 
                  dest="fake_implicit", default=False, 
                  help="just use a binary R as implicit?")
parser.add_option("-i", "--iterations", action="store", 
                  dest="iterations", type="string", default=None, 
                  help="number of iterations")

def main():
    # Get command line options
    options, args = parser.parse_args()
    prefix = options.prefix
    distance = options.distance
    neighbor_hood_size = int(options.k)
    l4 = int(options.l4)
    train = options.train
    cont = options.cont
    fake_implicit = options.fake_implicit
    n_iterations = int(options.iterations)

    if fake_implicit:
        faking = 'implicit-n'
    else:
        faking = 'implicit-y'

    # Parameters for the global model
    g = float(options.g)
    l5 = float(options.l5)

    # Get the ratings matrix
    R = np.array(pickle.load(open(pickle_directory + prefix + '_stars_train.p')).todense())
    N = np.array(R, copy=True)
    N[N.nonzero()] = 1

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

    # Get implicit ratings
    print '\nCalculating implicit data matrix ... '
    sys.stdout.flush()
    i_to_h = pickle.load(open(pickle_directory + prefix + '_idx_to_user_hash.p'))
    h_to_i = pickle.load(open(pickle_directory + prefix + '_user_hash_to_idx.p'))

    with open(data_directory + 'yelp_academic_dataset_user.json') as file:
        lc = 0
        for line in file:
            lc += 1

    with open(data_directory + 'yelp_academic_dataset_user.json') as file:
        i = 0
        for line in file:
            current = json.loads(line)
            if current['user_id'] in h_to_i:
                try:
                    friends = current['friends']
                    for f in friends:
                        if f in h_to_i:
                            N[h_to_i[current['user_id']],:] += R[h_to_i[f],:]
                except:
                    pass
            if i % 10 == 0 or i == (lc - 1):
                sys.stdout.write('\r' + str(round(float(i)/(lc - 1) * 100, 2)) + '%    ')
                sys.stdout.flush()
            i += 1

    N[N.nonzero()] = 1
    print 'Done'


    # Bin implicit
    print 'Binning implicit data ... ',
    sys.stdout.flush()
    user_implicit_buckets = {}
    for i in range(0,N.shape[0]):
        businesses = np.where(N[i,:] > 0)[0]
        user_implicit_buckets[i] = businesses
    print 'Done'

    if fake_implicit:
        N = np.array(R, copy=True)
        user_implicit_buckets = user_buckets

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


    # Get the Ratings
    R = pickle.load(open(pickle_directory + prefix + '_stars_train.p'))

    R_row = R.nonzero()[0]
    R_col = R.nonzero()[1]

    print "\nTraining ..."
    if train:
        if cont:
            P = pickle.load(open(pickle_directory + prefix + '_P_' + distance + '_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l4) + '.p'))
            W = pickle.load(open(pickle_directory + prefix + '_W_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l5) + '_g' + str(g) + '.p'))
            C = pickle.load(open(pickle_directory + prefix + '_C_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l5) + '_g' + str(g) + '.p'))
            P = np.array(P.todense())
            C = np.array(C.todense())
            W = np.array(W.todense())
        else:
            P = pickle.load(open(pickle_directory + prefix + '_P_' + distance + '_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l4) + '.p'))
            P = np.array(P.todense())
            C = np.array(P, copy = True)
            W = np.array(P, copy = True)
    else:
        try:
            print pickle_directory + prefix + '_P_' + distance + '_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l4) + '.p'
            P = pickle.load(open(pickle_directory + prefix + '_P_' + distance + '_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l4) + '.p'))
            W = pickle.load(open(pickle_directory + prefix + '_W_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l5) + '_g' + str(g) + '.p'))
            C = pickle.load(open(pickle_directory + prefix + '_C_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l5) + '_g' + str(g) + '.p'))
            P = np.array(P.todense())
            C = np.array(C.todense())
            W = np.array(W.todense())
        except:
            print "\nThose weights were not found. Are you sure you don't want to train?"
            sys.exit()

    if train:
        # Iterate and update the weights for explicit and implicit
        for iteration in range(n_iterations):
            sse = 0
            sse_n = 0

            print 'Iteration ' + str(iteration + 1)

            for k, u in enumerate(R_row):
                i = R_col[k]
                
                sigma_r = 0
                sigma_n = 0

                intersection1 = np.intersect1d(np.where(P[i,:] != 0)[0], user_buckets[u])
                
                for j in intersection1:
                    sigma_r += (R[u, j] - (mu + b_i[j] + b_u[u])) * W[i, j]

                intersection2 = np.intersect1d(np.where(P[i,:] != 0)[0], user_implicit_buckets[u])

                for j in intersection2:
                    sigma_n += C[i, j]

                if len(intersection1) == 0:
                    pow1 = 0
                else:
                    pow1 = np.power(len(intersection1), -0.5)

                if len(intersection2) == 0:
                    pow2 = 0
                else:
                    pow2 = np.power(len(intersection2), -0.5)

                r_hat = mu + b_u[u] + b_i[i] + pow1 * sigma_r + pow2 * sigma_n
                
                err = R[u, i] - r_hat
                sse += np.power(err, 2)
                sse_n += 1

                b_u[u] += g * (err - l5 * b_u[u])
                b_i[i] += g * (err - l5 * b_i[i])

                for j in intersection1:
                    W[i, j] += g * (np.power(len(intersection1), -0.5) * err * (R[u, j] - (mu + b_u[u] + b_i[j])) - l5 * W[i, j])

                for j in intersection2:
                    C[i, j] += g * (np.power(len(intersection2), -0.5) * err - l5 * C[i, j])
                
                if k % 10 == 0 or k == (len(R_row) - 1):
                    sys.stdout.write('\r' + str(round(float(k)/(len(R_row) - 1) * 100, 2)) + '%    ')
                    sys.stdout.flush()
            print '... RMSE = ' + str(np.sqrt(sse/sse_n)) + '\n'


    # Only keep k-most similar
    for i in range(n):
        modifier = np.zeros(n)
        modifier[np.argsort(W[i,:])[::-1][:neighbor_hood_size]] = 1
        W[i, :] = W[i, :] * modifier
        C[i, :] = C[i, :] * modifier

    if train:
        # Store similarities
        W = scipy.sparse.lil_matrix(W)
        C = scipy.sparse.lil_matrix(C)
        pickle.dump(W, open(pickle_directory + prefix + '_W_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l5) + '_g' + str(g) + '.p', "wb"))
        pickle.dump(C, open(pickle_directory + prefix + '_C_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l5) + '_g' + str(g) + '.p', "wb"))
        print 'Stored W matrix into ' + pickle_directory + prefix + '_W_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l5) + '_g' + str(g) + '.p'
        print 'Stored C matrix into ' + pickle_directory + prefix + '_C_k' + str(neighbor_hood_size) + '_' + faking + '_l' + str(l5) + '_g' + str(g) + '.p'
        W = np.array(W.todense())
        C = np.array(C.todense())

    # Clean the weights
    W[W < 0] = 0
    C[C < 0] = 0
    print 'Done'

    # Get the validation matrix
    V = pickle.load(open(pickle_directory + prefix + '_stars_validate.p'))
    R = R.todense()

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
        
        sigma_r = 0
        sigma_n = 0

        intersection1 = np.intersect1d(np.where(P[i,:] != 0)[0], user_buckets[u])

        for j in intersection1:
            sigma_r += (R[u, j] - (mu + b_i[j] + b_u[u])) * W[i, j]

        intersection2 = np.intersect1d(np.where(P[i,:] != 0)[0], user_implicit_buckets[u])

        for j in intersection2:
            sigma_n += C[i, j]

        if len(intersection1) == 0:
            pow1 = 0
        else:
            pow1 = np.power(len(intersection1), -0.5)

        if len(intersection2) == 0:
            pow2 = 0
        else:
            pow2 = np.power(len(intersection2), -0.5)

        r_hat = mu + b_u[u] + b_i[i] + pow1 * sigma_r + pow2 * sigma_n


        # Insert the prediction
        R[u,i] = r_hat

        if k % 10 == 0 or k == (len(required_row) - 1):
            sys.stdout.write('\r' + str(round(float(k)/(len(required_row) - 1) * 100, 2)) + '%    ')
            sys.stdout.flush()
    print 'Done'

    # Get the RMSE
    R = scipy.sparse.csr_matrix(R)

    print '\nRMSE for global: \t' + str(tools.rmse(R, V))

if __name__ == '__main__':
    main()
