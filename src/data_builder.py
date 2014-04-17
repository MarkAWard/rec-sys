import sys
from optparse import OptionParser
import json
from json.decoder import WHITESPACE
import operator
import cPickle as pickle
import Filter as Flt
import numpy as np
import pandas as pd

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

def x_builder(data_type, fields):
      lookup = pickle.load(open('../../' + data_type + 'id_to_indx.p', "rb" ))

      column_names = []
      for f in fields:
            field_split = f.split('.')
            if len(field_split) == 1:
                  column_names.append(field_split[0].lower())
            else:
                  column_names.append(field_split[1].lower())

      x = pd.DataFrame(columns=column_names)

      with open('../../yelp_academic_dataset_' + data_type + '.json') as file:
            for line in file:
                  to_save = {}
                  current = json.loads(line)
                  if current[data_type + '_id'] in lookup:
                        for f in fields:
                              field_split = f.split('.')
                              if len(field_split) == 1:
                                    try:
                                          to_save[field_split[0].lower()] = current[field_split[0].replace('__', ' ')]
                                    except:
                                          pass
                              else:
                                    try:
                                          to_save[field_split[1].lower()] = current[field_split[0].replace('__', ' ')][field_split[1].replace('__', ' ')]
                                    except:
                                          pass
                        to_save = pd.DataFrame([to_save, ])
                        to_save.index = [lookup[current[data_type + '_id']]]
                        x = x.append(to_save, ignore_index=True)

      return x

def main():
      options, args = parser.parse_args()

      x = x_builder(options.data_type, args)

      if options.create_pickle:
            with open(options.matrix_file, "wb") as outfp:
                  pickle.dump(x, outfp)
                  print 'Pickled ' + options.data_type + ' data matrix into: ' + options.matrix_file

if __name__ == "__main__":
    main()