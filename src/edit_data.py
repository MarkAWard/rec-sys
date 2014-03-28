"""
So far the only thing up and running is creating dictionaries used for 
lookups later. Next to add, a way to explore the data easily.

-----Creating Lookup Dictionaries-----

To create a pair of dictionaries use the flag:
 -d or --dictionary.

The script requires the path to the file with the json objects that will
 be used. To do so use -f or --file to specify the file:
 -f ../../file_with_json_obj.json

Specify which attribute(s) will be used to create the dictionary. There 
are two sets of flags:
 -a --attr
 -b --attr2
If --attr2 is not given, the default is to use 'review_count'

With only these options set the script will print both dictionaries 
to STDOUT. Can be used to make sure everything is working correctly 
before saving the objects

If you want the two dictionaries that are created to be saved with 
cPickle then use the flag:
 -p or --pickle

To specify where the pickles objects are saved, instead of using the defualt, 
there is a flag for each file name for each dictionary. With id corresponding
to the values obtained by --attr and index corresponds to values from --attr2
after descending sort. Naming convention should be clear:
 --id_to_index
 --indx_to_id

examples:
 Create dictionaries for user_id from data_file and pickle each in these seperate files
 python edit_data.py -d -attr user_id -f ../data_file.json -p --id_to_indx userid_to_indx.p --indx_to_id indx_to_userid.p
 Same as above
 python edit_data.py -d -attr user_id -attr2 review_count  -f ../data_file.json -p --id_to_indx userid_to_indx.p --indx_to_id indx_to_userid.p

 Create dictionaries for business_id from a file and print them to STDOUT
 python edit_data.py -d -attr business_id -f ../limited_data_file.json 


"""

import sys
from optparse import OptionParser
import json
from json.decoder import WHITESPACE
import operator
import cPickle as pickle

# command line parser
usage = "<script> [args ...]"
description = "Script for manipulating and exploring data files."
parser = OptionParser(usage=usage, description=description)
parser.add_option("-f", "--file", action="store", \
                      dest="file_path", type="string", default=None, \
                      help="path to data file", metavar="FILE")
parser.add_option("-o", "--output_file", action="store", \
                      dest="out_file", type="string", default=None, \
                      help="file to put output in", metavar="FILE")
parser.add_option("-a", "--attr", action="store", 
                  dest="attr", type="string", default=None,
                  help="Attribute name used for filtering")
parser.add_option("-b", "--attr2", action="store", 
                  dest="attr2", type="string", default=None,
                  help="Secondary attribute name")
parser.add_option("-p", "--pickle", action="store_true", 
                  dest="create_pickle", default=False,
                  help="pickle the output object")
parser.add_option("-d", "--dictionary", action="store_true", \
                      dest="create_dict", default=False, \
                      help="create a lookup dictionary")
parser.add_option("--id_to_indx", action="store", \
                      dest="id_to_indx", type="string", default=None, \
                      help="file for id to index dict", metavar="FILE")
parser.add_option("--indx_to_id", action="store", \
                      dest="indx_to_id", type="string", default=None, \
                      help="file for index to id dict", metavar="FILE")


# generator function to parse one json object at a time
def iterload(string_or_fp, cls=json.JSONDecoder, **kwargs):
    if isinstance(string_or_fp, file):
        string = string_or_fp.read()
    else:
        string = str(string_or_fp)
    # set args for decoder if specified
    decoder = cls(**kwargs)
    # skip over whitespace to beginning of first obj
    idx = WHITESPACE.match(string, 0).end()
    while idx < len(string):
        # object and index of the end of obj in string
        obj, end = decoder.raw_decode(string, idx)
        yield obj
        # skip whitespace till next object
        idx = WHITESPACE.match(string, end).end()

#
# ONLY for businesses file right now!!
#
def create_lookup_dict(infile, attr, attr2, id_to_indx=None, indx_to_id=None, pick=False):
    
    # if output files were not specified and pickling, set them
    if pick:
        if id_to_indx == None:
            id_to_indx = "id_to_indx.p" 
        if indx_to_id == None:
            indx_to_id = "indx_to_id.p" 
    # set attr2 defualt if None given
    if attr2 == None:
        attr2 = "review_count"

    # lookup dictionary to hold id's and index number
    lookup = {}
    with open(infile, "r") as fp:
        # iterate through objects in file
        for obj in iterload(fp):
            # make sure a valid attribute name was given
            try:
                attr_value = obj[attr]
            except KeyError:
                print "ERROR: %s, is not a valid attribute name " %attr
                exit()
            try:
                attr2_value = obj[attr2]
            except KeyError:
                print "ERROR: %s, is not a valid attribute name " %attr2
                exit()
            # upadate dictionary
            if attr_value not in lookup:
                lookup[ attr_value ] = attr2_value
            else:
                print attr_value + " already in lookup dictionary. Skipping duplicate" 
        
    # sort by descending review count in to a list
    sorted_list = sorted(lookup.iteritems(), key=operator.itemgetter(1), reverse=True)
    
    # change the value in lookup to be index from sorted list
    for item, indx in zip(sorted_list, xrange(len(sorted_list))):
        lookup[item[0]] = indx

    # dict for looking up id from an index
    lookup2 = {y:x for x,y in lookup.iteritems()}

    # Pickle it
    if pick:
        with open(id_to_indx, "wb") as outfp:
            pickle.dump(lookup, outfp)
            print "Pickled id --> index dictionary into: " + id_to_indx
        with open(indx_to_id, "wb") as outfp:
            pickle.dump(lookup2, outfp)
            print "Pickled index --> id dictionary into: " + indx_to_id
    # print to test that everything is working
    else:
        print(json.dumps(lookup, indent=3))
        print()
        print(json.dumps(lookup2, indent=3))

def main():
    options, args = parser.parse_args()
    
    if options.create_dict:
        if options.file_path != None and options.attr != None:
            print "creating a lookup dictionary from " + options.file_path 
            create_lookup_dict(options.file_path, \
                               options.attr, options.attr2, \
                               id_to_indx=options.id_to_indx, \
                               indx_to_id=options.indx_to_id, \
                               pick=options.create_pickle)
        else:
            print "ERROR: Must specify an input file and atleast one attribute to create a lookup dictionary"

    exit()

if __name__ == "__main__":
    main()

