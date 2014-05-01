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

    # method = options.method
    method = 'item-item'
    prefix = options.prefix

    data_stars_train, data_stars_validate, data_stars_test = tools.import_pickles(prefix, 'stars')
    data_business = pickle.load(open(pickle_directory + prefix + '_business.p'))
    data_business = np.array(data_business)
    data_stars_train = np.array(data_stars_train.todense())

    required_row = []
    required_col = []

    for i, s in enumerate(data_stars_validate.data):
        required_row.append(data_stars_validate.nonzero()[0][i])
        required_col.append(data_stars_validate.nonzero()[1][i])

    # print required_row[0:20]
    # print required_col[0:20]


    print_nobreak('Counting and binning users ... ')
    user_buckets = {}
    for i in range(0,data_stars_train.shape[0]):
        businesses = np.where(data_stars_train[i,:] > 0)[0]
        user_buckets[i] = businesses
    print 'Done'

    print_nobreak('Counting and binning businesses ... ')
    business_buckets = {}
    for i in range(0,data_stars_train.shape[1]):
        users = np.where(data_stars_train[:,i] > 0)[0]
        business_buckets[i] = users
    print 'Done'

    user_means = np.ma.average(data_stars_train, axis=1, weights=data_stars_train!=0).data
    #business_sims = np.zeros(10*data_stars_train.shape[1]).reshape(data_stars_train.shape[1], 10)

    num_b = data_stars_train.shape[1]
    # num_b = 10
    neighborhood_size = 30

    business_sims = np.zeros(neighborhood_size*num_b).reshape(num_b, neighborhood_size)
    business_tops = np.zeros(neighborhood_size*num_b).reshape(num_b, neighborhood_size)-1

    print 'Building adjusted cosine similarity matrix ... '
    for i in range(0,num_b):
        sims = {}
        for j in range(0, data_stars_train.shape[1]):
            num = 0
            denom_l = 0
            denom_r = 0
            ic = 0
            if j != i:
                for k in business_buckets[i]:
                    rating_j = data_stars_train[k, j]
                    if rating_j > 0:
                        rating_i = data_stars_train[k, i] 
                        mean = user_means[k]
                        num += (rating_j - mean) * (rating_i - mean)
                        denom_l += np.power(rating_j - mean, 2)
                        denom_r += np.power(rating_i - mean, 2)
                        ic += 1
            if ic > 1:
                #business_sims[i][j] = float(num)/(np.sqrt(denom_l) * np.sqrt(denom_r))
                #business_sims[i][j] = ic
                #if round(float(num)/(np.sqrt(denom_l) * np.sqrt(denom_r)), 5) != 1:
                sims[j] = 0 if np.isnan(round(float(num)/(np.sqrt(denom_l) * np.sqrt(denom_r)), 5)) else round(float(num)/(np.sqrt(denom_l) * np.sqrt(denom_r)), 5)
                # sims[j] = ic

        tops = sorted(sims, key=sims.get, reverse=True)[:neighborhood_size]
        for m, t in enumerate(tops):
            business_sims[i,m] = sims[t]
            business_tops[i,m] = t
        # business_sims[i] = sorted(sims, key=sims.get, reverse=True)[:10]
        # business_sims[i] = sorted(sims, key=sims.get, reverse=True)[:4]
        if i % 10 == 0 or i == (num_b - 1):
            sys.stdout.write('\r' + str(round(float(i)/(num_b - 1) * 100, 2)) + '%    ')
            sys.stdout.flush()
    print 'Done'

    # # print business_sims
    # # print business_tops

    # d = np.array(data_stars_train.todense())
    # # d = d[:,0:200]

    # print_nobreak('Counting and binning users ... ')
    # user_buckets = {}
    # for i in range(0,d.shape[0]):
    #     businesses = np.where(d[i,:] > 0)[0]
    #     user_buckets[i] = businesses
    # print 'Done'

    # print_nobreak('Counting and binning businesses ... ')
    # business_buckets = {}
    # for i in range(0,d.shape[1]):
    #     users = np.where(d[:,i] > 0)[0]
    #     business_buckets[i] = users
    # print 'Done'

    # d[np.isnan(d)] = 0
    # m = np.ma.average(d, axis=1, weights=d!=0).data
    # w = d!= 0
    # w = w.astype(int)
    # lol = w * m[:, np.newaxis]
    # z = d - lol
    # zm = np.ma.masked_array(z)
    # zm[zm==0] = np.ma.masked
    # zm_cosine = 1-pairwise_distances(zm.T, metric="cosine")
    # zm_cosine[np.isnan(zm_cosine)] = -3
    # np.fill_diagonal(zm_cosine, -4)

    # sim_matrix = zm_cosine




    # data_business[np.isnan(data_business)] = 0
    # b_cosine = 1-pairwise_distances(data_business, metric="cosine")

    # sim_matrix = b_cosine



    # business_sims = -1 * np.sort(-sim_matrix)[:,0:500]
    # business_tops = np.argsort(-sim_matrix)[:,0:500]


    nz = 0
    al = 0
    for idx, i in enumerate(required_row):
        j = required_col[idx]

        num = 0
        denom = 0

        # print business_tops[j,:]
        # print user_buckets[i]

        for jdx, b in enumerate(business_tops[j,:]):
            if b in user_buckets[i]:
                i = int(i)
                jdx = int(jdx)
                j = int(j)
                b = int(b)
                print "business_sims[" + str(j) + ", " + str(jdx) + "] = " + str(business_sims[j,jdx]) + " * data_stars_train[" + str(i) + ", " + str(b) + "] = " + str(data_stars_train[i,b])
                print "np.abs(business_sims[" + str(j) + ", " + str(jdx) + "] = " + str(np.abs(business_sims[j,jdx]))
                num += business_sims[j,jdx] * data_stars_train[i,b]
                denom += np.abs(business_sims[j,jdx])

        try:
            p = num/denom
        except:
            p = 0

        if p != 0:
            nz += 1

        data_stars_train[i,j] = p
        al += 1
    
    print "Nonzero predictions = " + str(float(nz)/al)



    data_stars_train = scipy.sparse.csr_matrix(data_stars_train)

    print "RMSE for '" + method + "': " + str(tools.rmse(data_stars_train, data_stars_validate))

def print_nobreak(s):
    print s,
    sys.stdout.flush()

if __name__ == "__main__":
    main()