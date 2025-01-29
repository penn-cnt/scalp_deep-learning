import numpy as np
import pandas as PD
from sys import argv
from tqdm import tqdm

def make_windowed(DF,outpath,window_size=9):

    # Make an output object we can append to and pass to the user
    outDF     = []
    outref    = []
    blacklist = ['file', 't_start', 'yasa_prediction', 'target_age_erin', 'target_gender', 'target_epidx', 'target_epitype', 'target_epilat', 'uid']
    whitelist = np.setdiff1d(DF.columns,blacklist)
    outcols   = blacklist.copy()

    # Ensure the dataframe is structured correctly
    DF = DF.sort_values(by=['uid','file','t_start']).reset_index(drop=True)

    # Find the indices for each unique file and user
    uid_file_inds      = DF.groupby(['uid','file']).indices
    uid_file_inds_keys = list(uid_file_inds.keys())

    # Rename the output columns
    attention_columns = []
    for idx in range(window_size):
        attention_columns.extend([f'{val}_{idx:02d}' for val in whitelist])
    attention_columns = np.array(attention_columns)
    outcols.extend(attention_columns)
    
    # Loop over the group index keys
    for group_inds in tqdm(uid_file_inds_keys,total=len(uid_file_inds_keys),desc='Making attention datafrme'):

        # Get the group inds
        ginds = uid_file_inds[group_inds]

        # Only work with data that has a large enough time horizon for the window
        if ginds.size > window_size:

            # Get the windowed indices
            winds = np.lib.stride_tricks.sliding_window_view(ginds,window_size)
            
            # Work through the slices to make a new attention dataframe
            for irow in winds:

                # get the first slice
                DF_slice        = DF.loc[irow[0]]
                ref_slice       = DF_slice[blacklist]
                attention_slice = DF_slice[whitelist].values

                for idx,ii in enumerate(irow[1:]):
                    
                    # Grab the current slice
                    iDF = DF.loc[ii][whitelist].values

                    # Join the dataframe slices
                    attention_slice = np.concatenate((attention_slice,iDF))
                
                # Append to output object
                outDF.append(attention_slice)
                outref.append(ref_slice)

    # Make the output object with a smaller typing
    outDF = PD.DataFrame(outDF,columns=attention_columns,dtype='float32')
    refDF = PD.DataFrame(outref,columns=blacklist)
    outDF = PD.concat((refDF,outDF),axis=1,ignore_index=True)
    outDF.columns = outcols
    outDF.to_csv(outpath)

if __name__ == '__main__':

    # Read in the data
    DF = PD.read_pickle(argv[1])
    make_windowed(DF,argv[2])