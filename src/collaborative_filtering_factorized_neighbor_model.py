'''
Factorized neighbor model model
References: Factor in the Neighbors: Scalable and Accurate Collaborative Filtering (Koren 2008)

Will try to learn something...

Work in progress. Pretty broke now... didn't fix anything up. Produces nans, doesn't utilize flags, etc.
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
parser.add_option("-m", "--method", action="store", 
                  dest="method", type="string", default=None,
                  help="method used to replace missing ratings", metavar="FILE")
parser.add_option("-p", "--prefix", action="store", 
                  dest="prefix", type="string", default=None, 
                  help="prefix for the data files", metavar="FILE")

def main():
    options, args = parser.parse_args()
    if options.prefix == None:
        prefix = 'data_20u_1b'
    else:
        prefix = options.prefix

    print 'Starting ...'

    R = np.array(pickle.load(open(pickle_directory + prefix + '_stars_train.p')).todense())

    print_nobreak('Counting and binning users ... ')
    R_u = {}
    for i in range(0,R.shape[0]):
        businesses = np.where(R[i,:] > 0)[0]
        R_u[i] = businesses
    print 'Done'

    N_u = R_u

    n_iter = 5
 
    q, x, y, b_u, b_i = lfnm(R, 50, R_u, N_u, n_iter)


def lfnm(R, rank, R_u, N_u, n_iter):
    m = R.shape[0]      # number of users
    n = R.shape[1]      # number of businesses

    tsvd = TruncatedSVD(n_components = 50, n_iterations=10)
    transformer = tsvd.fit( R )
    q = np.array(transformer.components_).T
    print q.shape
    p = np.array(tsvd.fit_transform( R ))
    print p.shape

    # q = np.zeros(rank * n).reshape(n, rank)
    # p = np.zeros(rank * m).reshape(m, rank)
    x = np.zeros(rank * n).reshape(n, rank)
    y = np.zeros(rank * n).reshape(n, rank)
    #b_u = np.zeros(m)
    #b_i = np.zeros(n)
    b_u = np.ma.average(R, axis=1, weights=R!=0).data
    b_i = np.ma.average(R, axis=0, weights=R!=0).data

    g = 0.002
    l = 0.04
    
    mu = np.ma.average(R, weights=R!=0)

    for count in range(n_iter):
        for u in range(m):
            sigma = 0
            # print p[u,:].min()
            for j in R_u[u]:
                sigma += (R[u,j] - (mu + b_u[u] + b_i[j])) * x[j,:]
                #print "(" + str(R[u,j]) + " - " + str(mu) + " + " +  str(b_u[u]) + " + " + str(b_i[j]) + ") * "  + str(x[j,:].max())
            p[u, :] = np.power(len(R_u[u]), -0.5) + sigma
            # print sigma
            # print p[u,:].min()
            sigma = 0
            for j in N_u[u]:
                sigma += y[j,:]
            p[u, :] += np.power(len(N_u[u]), -0.5) + sigma

            # print p[u,:].min()
            # print '--------------'

            sum = 0
            for i in R_u[u]:
                r_hat = mu + b_u[u] + b_i[i] + np.dot(q[i,:], p[u,:])
                err = R[u, i] - r_hat
                
                sum += err * q[i,:]

                q[i,:] += g * (err * p[u,:] - l * q[i,:])
                b_u[u] += g * (err - l * b_u[u])
                b_i[i] += g * (err - l * b_i[i])

            for i in R_u[u]:
                x[i,:] += g * (np.power(len(R_u[u]), -0.5) * (R[u,i] - (mu + b_u[u] + b_i[i])) * sum - l * x[i,:])

            for i in N_u[u]:
                y[i,:] += g * (np.power(len(N_u[u]), -0.5) * sum - l * y[i,:])
        print "Iteration " + str(count + 1) + "/" + str(n_iter) + " ended."
    return q, x, y, b_u, b_i

def print_nobreak(s):
    print s,
    sys.stdout.flush()

if __name__ == '__main__':
    main()