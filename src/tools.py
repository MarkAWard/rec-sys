"""
Tools for analyzing Yelp data

Usage:
Can be used in any python file in src by including
import tools

Functions:
* import_pickles(prefix, type)
Used to import pickles, use prefix to math what was used in data_builder.py and select a type of
data: user, business, stars, text, lookup. The function will return a different number of files
depending on the 'type' passed. See function for details.

* rmse(predicted, truth)
Used to get the rmse given two sparse data matrices. The returned value will only text against
non-zero values that exist in the truth and predicted matrix. The truth matrix will almost always
be either the validate or test matrix.
"""

import cPickle as pickle
import os
import scipy.sparse
import math
import numpy as np

def import_pickles(prefix, type):
    # pickle directory
    pickle_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/pickle/')
    # data directory
    data_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/data/')

    if type == 'user':
        data_user = pickle.load(open(pickle_directory + prefix + '_user.p'))
        return data_user

    elif type == 'business':
        data_business = pickle.load(open(pickle_directory + prefix + '_business.p'))
        return data_business

    elif type == 'stars':
        data_stars_train = pickle.load(open(pickle_directory + prefix + '_stars_train.p'))
        data_stars_validate= pickle.load(open(pickle_directory + prefix + '_stars_validate.p'))
        data_stars_test = pickle.load(open(pickle_directory + prefix + '_stars_test.p'))
        return data_stars_train, data_stars_validate, data_stars_test

    elif type == 'text':
        data_text_train = pickle.load(open(pickle_directory + prefix + '_text_train.p'))
        data_text_validate= pickle.load(open(pickle_directory + prefix + '_text_validate.p'))
        data_text_test = pickle.load(open(pickle_directory + prefix + '_text_test.p'))
        return data_text_train, data_text_validate, data_text_test

    elif type == 'lookup':
        lookup_business_hash_to_idx = pickle.load(open(pickle_directory + prefix + '_business_hash_to_idx.p'))
        lookup_idx_to_business_hash = pickle.load(open(pickle_directory + prefix + '_idx_to_business_hash.p'))
        lookup_idx_to_user_hash = pickle.load(open(pickle_directory + prefix + '_idx_to_user_hash.p'))
        lookup_user_hash_to_idx = pickle.load(open(pickle_directory + prefix + '_user_hash_to_idx.p'))
        return lookup_business_hash_to_idx, lookup_idx_to_business_hash, lookup_idx_to_user_hash, lookup_user_hash_to_idx

def rmse(predicted, truth):
    sum = 0
    n = 0
    
    for idx, t_s in enumerate(truth.data):
        i = truth.nonzero()[0][idx]
        j = truth.nonzero()[1][idx]

        try:
            p_s = predicted.data[predicted.indptr[i] + np.where(predicted.indices[predicted.indptr[i]:predicted.indptr[i+1]] == j)[0]][0]
            sum += math.pow(p_s - t_s, 2)
            n += 1
        except:
            p_s = 0
            sum += math.pow(p_s - t_s, 2)
            n += 1

    if n != 0:
        return math.sqrt(sum/n)
    else:
        print 'Returning None. No values could be compared. Are you sure the predictions were made?'
        return None