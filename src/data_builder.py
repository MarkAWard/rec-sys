"""
Create data frames from JSON data and lookup dictionaries.

This script can create two types of files
 (1) Feature matrix with users/businesses as rows, and user selected features as columns
 (2) User x Business matrix with star rating or review text in each cell

The following flags are used,

-u / -b
Utilize user or business data. Call both flags to create a user-business data matrix. When 
calling each flag, different files will be made:
 (1) -u will create a file containing a pandas dataframe (users as rows, fields as columns).
     Filename: prefix + _user.p
 (2) -b will create a file containing a pandas dataframe (businesses as rows, fields as columns).
     Filename: prefix + _business.p
 (3) -ub will create 7 files; 4 lookup dictionaries and 3 sparse data matrices. The lookups will
     match the data in the sparse matrices since due to cleaning, the original lookups may no 
     longer match.
     Filename: prefix + _business_hash_to_idx.p
     Filename: prefix + _idx_to_business_hash.p
     Filename: prefix + _idx_to_user_hash.p
     Filename: prefix + _user_hash_to_idx.p
     Filename: prefix + _stars_test.p
     Filename: prefix + _stars_train.p
     Filename: prefix + _stars_validate.p

--user_lookup
Filename of the user_lookup file, must be hash -> idx.

--business_lookup
Filename of the business_lookup file, must be hash -> idx.

--prefix
Filenames are already chosen via the method selected (-u or -b). This prefix will be added to
the beginning of all filenames.


When creating a matrix of type
 (1) A list of features must be listed as arguments in the same format they were written in the 
     original JSON files. The only exception is that spaces must be written as an underscore (_).
     If you want a feature that is within another feature, use a dot (.) to seperate them. For 
     example, businesses have many fields in the 'attributes' field, you can get the 'Has TV'
     feature by asking for 'attributes.Has_TV'. For fields that are categorical, it will make
     a new feature for each category, for example 'Good For' can take on values 'dessert', 'dinner',
     'snacks', etc. Asking for this feature will create features 'goodfor_dessert', 'goodfor_dinner',
     'goodfor_snacks', etc. Boolean values are coded as 0 (False) or 1 (True)
 (2) No modifications


Example call to create a user data frame with supplied fields:
python data_builder.py -u --user_lookup data_5u_10b_user_hash_to_idx.p --prefix data_5u_10b yelping_since votes review_count average_stars

Example call to create a business data frame with supplied fields:
python data_builder.py -b --business_lookup data_5u_10b_business_hash_to_idx.p --prefix data_5u_10b stars attributes.Parking

Example call to create a user + business sparse matrix:
python data_builder.py -ub --user_lookup 5_user_to_indx.p --business_lookup 10_bus_to_indx.p -p --prefix data_5u_10b
"""

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


# pickle directory
pickle_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/pickle/')
# data directory
data_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/data/')

# command line parser
usage = "<script> [args ...]"

description = "Script for building data matrix objects."

parser = OptionParser(usage=usage, description=description)
parser.add_option("-u", "--user", action="store_true", 
                  dest="data_type_user", default=False,
                  help="use the user data")
parser.add_option("-b", "--business", action="store_true", 
                  dest="data_type_business", default=False,
                  help="use the business data")
parser.add_option("-a", "--arrays", action="store_false", 
                  dest="create_matrix", default=True,
                  help="arrays instead of sparse matrix")
parser.add_option("-p", "--pickle", action="store_true", 
                  dest="create_pickle", default=False,
                  help="pickle the output object")
parser.add_option("--user_lookup", action="store", 
                  dest="user_lookup", type="string", default=None, 
                  help="file for data matrix", metavar="FILE")
parser.add_option("--business_lookup", action="store", 
                  dest="business_lookup", type="string", default=None, 
                  help="file for data matrix", metavar="FILE")
parser.add_option("--prefix", action="store", 
                  dest="prefix", type="string", default=None, 
                  help="prefix for file names")
parser.add_option("--no_plot", action="store_false", 
                  dest="plot", default=True, 
                  help="No plotting")

def data_x_builder(data_type, lookup_file, prefix, fields):
    lookup = pickle.load(open(pickle_directory + lookup_file, "rb" ))

    x = pd.DataFrame()

    with open(data_directory + 'yelp_academic_dataset_' + data_type + '.json') as file:
        for line in file:
            to_save = {}
            current = json.loads(line)
            if current[data_type + '_id'] in lookup:
                for f in fields:
                    field_split = f.split('.')
                    if len(field_split) == 1:
                        df_field = re.sub('[_\-\s]', '', field_split[0].lower())
                        if df_field not in ['votes']:
                            try:
                                to_save[df_field] = int(current[field_split[0]]) if type(current[field_split[0]]) == bool else current[field_split[0]]
                            except:
                                pass
                        else:
                            try:
                                for key, value in current[field_split[0]].iteritems():
                                    save_key = df_field + '_' + key
                                    to_save[save_key] = int(value) if type(value) == bool else value
                            except:
                                pass
                    else:
                        df_field = re.sub('[_\-\s]', '', field_split[1].lower())
                        od_field = re.sub('_', ' ', field_split[1])
                        if df_field not in ['ambience', 'goodfor', 'parking', 'music']:
                            try:
                                to_save[df_field] = int(current[field_split[0]][od_field]) if type(current[field_split[0]][od_field]) == bool else current[field_split[0]][od_field]
                            except:
                                pass
                        else:
                            try:
                                for key, value in current[field_split[0]][od_field].iteritems():
                                    save_key = df_field + '_' + key
                                    to_save[save_key] = int(value) if type(value) == bool else value
                            except:
                                pass


                to_save = pd.DataFrame([to_save, ])
                to_save.index = [lookup[current[data_type + '_id']]]
                x = x.append(to_save, ignore_index=True)
    
    print x.ix[0:4,]
    
    pickle_object(x, prefix + '_' + data_type + '.p', data_type + 'data frame')


def data_xy_builder(lookup_file_1, lookup_file_2, prefix, matrix=True, plot=True):
    x_lookup = pickle.load(open(pickle_directory + lookup_file_1, "rb" ))
    y_lookup = pickle.load(open(pickle_directory + lookup_file_2, "rb" ))

    x = sparse_review_builder('fre', 'stars', x_lookup, y_lookup)

    user_sum = np.array(x.sum(1)).reshape(1, x.shape[0])[0]
    business_sum = np.array(x.sum(0)).reshape(1, x.shape[1])[0]

    print "Before cleaning statistics"
    print "Number of users: " + str(len(x_lookup))
    print "Smallest number of reviews for a user: " + str(user_sum.min())
    print "Number of businesses : " + str(len(y_lookup))
    print "Smallest number of reviews for a business: " + str(business_sum.min())
    print "\n"

    user_keep = np.where(user_sum >= 5)[0]
    business_keep = np.where(business_sum >= 1)[0]

    i = 0
    x_lookup_new = {}
    for key, value in x_lookup.iteritems():
        if value in user_keep:
            x_lookup_new[key] = i
            i += 1

    i = 0
    y_lookup_new = {}
    for key, value in y_lookup.iteritems():
        if value in business_keep:
            y_lookup_new[key] = i
            i += 1

    x_lookup = x_lookup_new
    y_lookup = y_lookup_new

    x_stars = sparse_review_builder('val', 'stars', x_lookup, y_lookup)
    # x_text = sparse_review_builder('val', 'text', x_lookup, y_lookup)

    user_sum = np.array(x_stars.sum(1)).reshape(1, x_stars.shape[0])[0]
    business_sum = np.array(x_stars.sum(0)).reshape(1, x_stars.shape[1])[0]

    print "After cleaning statistics"
    print "Number of users: " + str(len(x_lookup))
    print "Smallest number of reviews for a user: " + str(user_sum.min())
    print "Number of businesses : " + str(len(y_lookup))
    print "Smallest number of reviews for a business: " + str(business_sum.min())
    print "\n"

    if plot:
    
        plt.subplot(2,2,1)
        plt.hist(user_sum, user_sum.max())
        plt.title('Num. of Users (y-axis) that have x reviews (x-axis)')

        plt.subplot(2,2,2)
        plt.hist(user_sum, user_sum.max())
        plt.title('Num. of Users (y-axis) that have x reviews (x-axis) [truncated x-axis]')
        plt.xlim(0,50)

        plt.subplot(2,2,3)
        plt.hist(business_sum, business_sum.max())
        plt.title('Num. of Businesses (y-axis) that have x reviews (x-axis)')

        plt.subplot(2,2,4)
        plt.hist(business_sum, business_sum.max())
        plt.title('Num. of Businesses (y-axis) that have x reviews (x-axis) [truncated x-axis]')
        plt.xlim(0,50)
        plt.show()


    train_col = []
    train_row = []
    train_val_stars = []
    # train_val_text = []
    validate_col = []
    validate_row = []
    validate_val_stars = []
    # validate_val_text = []
    test_col = []
    test_row = []
    test_val_stars = []
    # test_val_text = []

    print "Partitioning data into train, validate, and test sets..."
    for i in range(0, x_stars.shape[0]):
        indices = np.random.permutation(np.where(x_stars.nonzero()[0] == i)[0])
        data_stars = x_stars.data[indices]
        # data_text = x_text.data[indices]
        length = len(indices)

        train_col = np.concatenate((train_col, x_stars.nonzero()[1][indices[:int(length * 0.8)]]))
        train_row = np.concatenate((train_row, indices[:int(length * 0.8)] * 0 + i))
        train_val_stars = np.concatenate((train_val_stars, data_stars[:int(length * 0.8)]))
        # train_val_text = np.concatenate((train_val_text, data_text[:int(length * 0.8)]))

        validate_col = np.concatenate((validate_col, x_stars.nonzero()[1][indices[int(length * 0.8):int(length * 0.9)]]))
        validate_row = np.concatenate((validate_row, indices[int(length * 0.8):int(length * 0.9)] * 0 + i))
        validate_val_stars = np.concatenate((validate_val_stars, data_stars[int(length * 0.8):int(length * 0.9)]))
        # validate_val_text = np.concatenate((validate_val_text, data_text[int(length * 0.8):int(length * 0.9)]))

        test_col = np.concatenate((test_col, x_stars.nonzero()[1][indices[int(length * 0.9):]]))
        test_row = np.concatenate((test_row, indices[int(length * 0.9):] * 0 + i))
        test_val_stars = np.concatenate((test_val_stars, data_stars[int(length * 0.9):]))
        # test_val_text = np.concatenate((test_val_text, data_text[int(length * 0.9):]))
        
        if i % 100 == 0 or i == (x_stars.shape[0] - 1):
            sys.stdout.write('\r' + str(round(float(i + 1)/x_stars.shape[0] * 100,2)) + '%')
            sys.stdout.flush()
        if i == (x_stars.shape[0] - 1):
            print " ... Done"


    # create sparse matrix object
    if matrix:
        x_train_stars = scipy.sparse.csr_matrix((train_val_stars,(train_row,train_col)), shape=(len(x_lookup),len(y_lookup)))
        x_validate_stars = scipy.sparse.csr_matrix((validate_val_stars,(validate_row,validate_col)), shape=(len(x_lookup),len(y_lookup)))
        x_test_stars = scipy.sparse.csr_matrix((test_val_stars,(test_row,test_col)), shape=(len(x_lookup),len(y_lookup)))

        # x_train_text = scipy.sparse.csr_matrix((train_val_text,(train_row,train_col)), shape=(len(x_lookup),len(y_lookup)))
        # x_validate_text = scipy.sparse.csr_matrix((validate_val_text,(validate_row,validate_col)), shape=(len(x_lookup),len(y_lookup)))
        # x_test_text = scipy.sparse.csr_matrix((test_val_text,(test_row,test_col)), shape=(len(x_lookup),len(y_lookup)))
        
        print '\n'
        print 'Saving objects to pickles...'
        pickle_object(x_train_stars, prefix + '_stars_train.p', 'stars train matrix')
        pickle_object(x_validate_stars, prefix + '_stars_validate.p', 'stars validate matrix')
        pickle_object(x_test_stars, prefix + '_stars_test.p', 'stars test matrix')
        # pickle_object(x_train_text, prefix + '_text_train.p', 'stars train matrix')
        # pickle_object(x_validate_text, prefix + '_text_validate.p', 'stars validate matrix')
        # pickle_object(x_test_text, prefix + '_text_test.p', 'stars test matrix')

    # do not create matrix, only save the array of indices and data
    else:
        x_train_stars = np.vstack( (train_row, train_col, train_val_stars) )
        x_validate_stars = np.vstack( (validate_row, validate_col, validate_val_stars) )
        x_test_stars = np.vstack( (test_row, test_col, test_val_stars) )

        print '\n'
        print 'Saving objects to pickles...'
        pickle_object(x_train_stars, prefix + '_arr_stars_train.p', 'stars train data array')
        pickle_object(x_validate_stars, prefix + '_arr_stars_validate.p', 'stars validate data array')
        pickle_object(x_test_stars, prefix + '_arr_stars_test.p', 'stars test data array')
        
    pickle_object(x_lookup, prefix + '_user_hash_to_idx.p', 'user hash to idx lookup')
    pickle_object({v:k for k, v in x_lookup.items()}, prefix + '_idx_to_user_hash.p', 'idx to user hash lookup')
    pickle_object(y_lookup, prefix + '_business_hash_to_idx.p', 'business hash to idx lookup')
    pickle_object({v:k for k, v in y_lookup.items()}, prefix + '_idx_to_business_hash.p', 'idx to business hash lookup')


def pickle_object(object, prefix, type):
    output_file = pickle_directory + prefix
    with open(output_file, "wb") as out:
        pickle.dump(object, out)
        print 'Pickled ' + type + ' into: ' + output_file

def clean(incoming):
    incoming = unicode(incoming)
    incoming = re.sub(r"\s+", ' ', incoming)
    incoming = incoming.strip().encode('utf-8').lower()
    return incoming

def sparse_review_builder(method, type, x_lookup, y_lookup):
    row = []
    col = []
    val = []
    fre = []

    with open(data_directory + 'yelp_academic_dataset_review.json') as file:
        #i = 0
        for line in file:
            current = json.loads(line)
            
            if current['user_id'] in x_lookup and current['business_id'] in y_lookup:
                row.append(x_lookup[current['user_id']])
                col.append(y_lookup[current['business_id']])
                val.append(clean(current[type]))
                fre.append(1)
    if type == 'stars':
        val = np.array(val, dtype='int')

    if method == 'val':
        x = scipy.sparse.csr_matrix((val,(row,col)), shape=(len(x_lookup),len(y_lookup)))
        if type == 'stars':
            x_fre = scipy.sparse.csr_matrix((fre,(row,col)), shape=(len(x_lookup),len(y_lookup)))
            return x/x_fre
        else:
            return x
    elif method == 'fre':
        x_fre = scipy.sparse.csr_matrix((fre,(row,col)), shape=(len(x_lookup),len(y_lookup)))
        return x_fre

def main():
    options, args = parser.parse_args()

    data_type_user = options.data_type_user
    data_type_business = options.data_type_business
    create_matrix = options.create_matrix
    prefix = options.prefix
    plot = options.plot

    if data_type_user and data_type_business:
        print 'Building user + business matrices'
        lookup_file_1 = options.user_lookup
        lookup_file_2 = options.business_lookup
        data_xy_builder(lookup_file_1, lookup_file_2, prefix, matrix=create_matrix, plot=plot)
    
    elif data_type_user:
        print 'Building user data frame...'
        lookup_file = options.user_lookup
        data_x_builder('user', lookup_file, prefix, args)
    
    elif data_type_business:
        print 'Building business data frame...'
        lookup_file = options.business_lookup
        data_x_builder('business', lookup_file, prefix, args)


if __name__ == "__main__":
    main()
