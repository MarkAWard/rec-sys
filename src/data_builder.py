"""
Create data frames from JSON data and lookup dictionaries.

This script can create two types of files
 (1) Feature matrix with users/businesses as rows, and user selected features as columns
 (2) User x Business matrix with star rating or review text in each cell

The following flags are used,

-d [user|business|user+business]
When used with user or business, a data frame of type (1) listed above will be created, when
used with user+business, the second type (2) will be created

-p
Set if a pickle object should be saved

--matrix_file
Must be set if -p is specified. Given a file name, a pickle will be saved in the pickle folder

When creating a matrix of type
 (1) a list of features must be listed as arguments in the same format they were written in the 
     original JSON files. The only exception is that spaces must be written as an underscore (_).
     If you want a feature that is within another feature, use a dot (.) to seperate them. For 
     example, businesses have many fields in the 'attributes' field, you can get the 'Has TV'
     feature by asking for 'attributes.Has_TV'. For fields that are categorical, it will make
     a new feature for each category, for example 'Good For' can take on values 'dessert', 'dinner',
     'snacks', etc. Asking for this feature will create features 'goodfordessert', 'goodfordinner',
     'goodforsnacks', etc.
 (2) must take the value 'text' or 'stars'
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


# pickle directory
pickle_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/pickle/')
# data directory
data_directory = os.path.expanduser('~/Dropbox/yelp-rec-sys/data/')

# command line parser
usage = "<script> [args ...]"

description = "Script for building data matrix objects."

parser = OptionParser(usage=usage, description=description)
parser.add_option("-d", "--data", action="store", 
                  dest="data_type", type="string", default=None, 
                  help="type of data")
parser.add_option("-p", "--pickle", action="store_true", 
                  dest="create_pickle", default=False,
                  help="pickle the output object")
parser.add_option("--matrix_file", action="store", 
                  dest="matrix_file", type="string", default=None, 
                  help="file for data matrix", metavar="FILE")

def data_x_builder(data_type, fields):
    lookup = pickle.load(open(pickle_directory + data_type + 'id_to_indx.p', "rb" ))

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
                        try:
                            to_save[df_field] = int(current[field_split[0]]) if type(current[field_split[0]]) == bool else current[field_split[0]]
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
                                    save_key = df_field + key
                                    to_save[save_key] = int(value) if type(value) == bool else value
                            except:
                                pass


                to_save = pd.DataFrame([to_save, ])
                to_save.index = [lookup[current[data_type + '_id']]]
                x = x.append(to_save, ignore_index=True)
    print x.ix[0:4,]
    return x

def data_xy_builder(data_type_1, data_type_2, fields):
    x_lookup = pickle.load(open(pickle_directory + 'userid_to_indx.p', "rb" ))
    y_lookup = pickle.load(open(pickle_directory + 'businessid_to_indx.p', "rb" ))

    row = []
    col = []
    val = []

    type = fields[0]

    with open(data_directory + 'yelp_academic_dataset_review.json') as file:
        for line in file:
            current = json.loads(line)
            try:
                row.append(x_lookup[current['user_id']])
                col.append(y_lookup[current['business_id']])
                val.append(clean(current[type]))
            except:
                pass

    x = scipy.sparse.coo_matrix((val,(row,col)), shape=(len(x_lookup),len(y_lookup)))

    return x

def clean(incoming):
    incoming = unicode(incoming)
    incoming = re.sub(r"\s+", ' ', incoming)
    incoming = incoming.strip().encode('utf-8').lower()
    return incoming

def main():
    options, args = parser.parse_args()

    data_type = options.data_type.split('+')

    if len(data_type) == 1:
        data_type = data_type[0]
        x = data_x_builder(data_type, args)
    else:
        data_type_1 = data_type[0]
        data_type_2 = data_type[1]
        data_type = data_type_1 + '+' + data_type_2
        x = data_xy_builder(data_type_1, data_type_2, args)

    pick = options.create_pickle

    if pick:
        output_file = pickle_directory + options.matrix_file

        with open(output_file, "wb") as out:
            pickle.dump(x, out)
            print 'Pickled ' + data_type + ' data matrix into: ' + output_file

if __name__ == "__main__":
    main()