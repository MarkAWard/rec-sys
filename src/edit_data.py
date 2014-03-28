from optparse import OptionParser
import json
from json.decoder import WHITESPACE
import operator
import cPickle as pickle

# command line parser
usage = "<script> flag [args ...]"
description = "I don't know how to use this yet either"
parser = OptionParser(usage=usage, description=description)
parser.add_option("-f", "--file", action="store", \
                      dest="file_path", type="string", default=None, \
                      help="path to data file", metavar="FILE")
parser.add_option("-o", "--output_file", action="store", \
                      dest="out_file", type="string", default=None, \
                      help="file to put output in", metavar="FILE")
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



def create_lookup_dict(infile, id_to_indx=None, indx_to_id=None, pick=False):
    
    # if output files were not specified and pickling, set them
    if pick:
        if id_to_indx == None:
            id_to_indx = "id_to_indx.p" 
        if indx_to_id == None:
            indx_to_id = "indx_to_id.p" 

    # lookup dictionary to hold id's and index number
    lookup = {}
    with open(infile, "r") as fp:
        for obj in iterload(fp):
            if obj['business_id'] not in lookup:
                lookup[ obj['business_id'] ] = obj['review_count']
            else:
                print obj['business_id'] + " already in lookup dictionary" 
        
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
        print lookup
        print lookup2

def main():
    options, args = parser.parse_args()
    print options
    print args
    
    if options.create_dict and options.file_path != None:
        print "creating a lookup dictionary from " + options.file_path 
        create_lookup_dict(options.file_path, \
                               id_to_indx=options.id_to_indx, \
                               indx_to_id=options.indx_to_id, \
                               pick=options.create_pickle)

    exit()

if __name__ == "__main__":
    main()

