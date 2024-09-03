import os
import re
import pwd
import time
import tqdm
import pickle
import argparse
import numpy as np
import pandas as PD

def find_targets(args):

    # Get all the target dictionaries
    target_files = []
    for dirpath, dirs, files in os.walk(args.rootdir):  
        for filename in files:
            fname = os.path.join(dirpath,filename) 
            if fname.endswith('targets.pickle'): 
                target_files.append(fname)
    return np.array(target_files).reshape((-1,1))

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--rootdir", type=str, required=True, help="Root directory to search within for target data.")
    parser.add_argument("--outfile", type=str, required=True, help="Path to save results to")
    args = parser.parse_args()

    # Get filepaths to each target file
    target_files = find_targets(args)

    # Make a dataframe we can just output
    candidate_DF        = PD.DataFrame(target_files,columns=['fpath'])
    
    # Add info we can get from just the bids path
    candidate_DF['uid']            = candidate_DF['fpath'].apply(lambda x:int(x.split('sub-')[1].split('/')[0]))
    candidate_DF['subject_number'] = candidate_DF['uid']
    candidate_DF['session_number'] = candidate_DF['fpath'].apply(lambda x:int(re.search(r'\d+$', x.split('ses-')[1].split('/')[0]).group()))
    candidate_DF['run_number']     = candidate_DF['fpath'].apply(lambda x:int(x.split('run-')[1].split('_')[0]))
    candidate_DF['source']         = 'ieeg.org'

    # Using OS, get additional info
    candidate_DF['gendate'] = candidate_DF['fpath'].apply(lambda x:time.strftime('%d-%m-%y', time.localtime(os.path.getctime(x))))
    candidate_DF['creator'] = candidate_DF['fpath'].apply(lambda x:pwd.getpwuid(os.stat(x).st_uid).pw_name)

    # Use the pickle info to get the remaining subject_map columns
    print("Reading in pickles. This may take awhile.")
    orig_filename = []
    start_sec     = []
    duration_sec  = []
    for ifile in tqdm.tqdm(candidate_DF.fpath.values, total=candidate_DF.shape[0],leave=False):
        idata = pickle.load(open(ifile,'rb'))
        orig_filename.append(idata['ieeg_file'])
        start_sec.append(idata['ieeg_start_sec'])
        duration_sec.append(idata['ieeg_duration_sec'])
    candidate_DF['orig_filename'] = orig_filename
    candidate_DF['start_sec']     = start_sec
    candidate_DF['duration_sec']  = duration_sec

    # Drop the fileppath columns
    candidate_DF = candidate_DF.drop(['fpath'],axis=1)

    # Sort the results
    candidate_DF.sort_values(by=['subject_number','session_number','run_number'],inplace=True)
    candidate_DF = candidate_DF[['orig_filename','source','creator','gendate','uid','subject_number','session_number','run_number','start_sec','duration_sec']]

    # Save the results
    candidate_DF.to_csv(args.outfile,index=False)