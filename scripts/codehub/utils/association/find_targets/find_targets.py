import os
import nltk
import pickle
import argparse
import numpy as np
import pandas as PD
from sys import exit
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('stopwords')

def return_tokens(istr):

    stop_words      = set(stopwords.words('english'))
    tokenizer       = RegexpTokenizer(r'\w+')
    tokens          = tokenizer.tokenize(istr.lower())
    filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return filtered_tokens

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--rootdir", type=str, required=True, help="Root directory to search within for target data.")
    parser.add_argument("--outfile", type=str, required=True, help="Path to save results to")
    args = parser.parse_args()

    # Get all the target dictionaries
    target_files = []
    for dirpath, dirs, files in os.walk(args.rootdir):  
        for filename in files:
            fname = os.path.join(dirpath,filename) 
            if fname.endswith('targets.pickle'): 
                target_files.append(fname)
    
    # Loop through each file and read in the target information. Then tokenize and store to final search dictionary
    lookup_dict = {}
    for ifile in target_files:
        target_dict = pickle.load(open(ifile,"rb"))
        annot_str   = target_dict['annotation']
        target_str  = target_dict['target']
        all_tokens  = return_tokens(annot_str)
        all_tokens  = all_tokens + return_tokens(target_str)
        
        # Loop over the tokens to add to the lookup
        for itoken in all_tokens:
            if itoken not in lookup_dict.keys():
                lookup_dict[itoken]          = {}
                lookup_dict[itoken]['count'] = 0
                lookup_dict[itoken]['files'] = []
            lookup_dict[itoken]['count'] += 1
            lookup_dict[itoken]['files'].append(ifile.replace('_targets.pickle','.edf'))

    # Make a pretty lookup table for the user
    index         = np.array(list(lookup_dict.keys())).reshape((-1,1))
    counts        = np.array([lookup_dict[ikey]['count'] for ikey in lookup_dict.keys()]).reshape((-1,1))
    DF            = PD.DataFrame(counts,columns=['count'])
    DF['keyword'] = index
    DF            = DF.sort_values(by=['count'],ascending=False)
    DF.set_index('keyword',drop=True,inplace=True)
    
    # Print the results
    print(DF)

    # Ask the user for which keyword to save the filepaths for
    token = input("Enter the keyword you want the file list for? (Q/q quit). ")
    if token.lower() == 'q':exit()
    while token.lower() not in index:
        token = input("Token not found. Please enter the keyword you want the file list for? (Q/q quit). ")
        if token.lower() == 'q':exit()
    
    # Save the files to the output file
    outfiles = lookup_dict[token]['files']
    fp       = open(args.outfile,'w')
    for ifile in outfiles:
        fp.write(f"{ifile}\n")
    fp.close()
