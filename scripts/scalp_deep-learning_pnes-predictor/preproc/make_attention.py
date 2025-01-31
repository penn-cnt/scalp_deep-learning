import yaml
import pickle
import numpy as np
import pandas as PD
import pylab as PLT
from tqdm import tqdm
from sys import argv,exit
from sklearn.decomposition import PCA

def make_windowed(DF,window_size=9):

    # Make an output object we can append to and pass to the user
    outDF     = []
    outref    = []
    blacklist = ['file', 't_start', 'uid', 'target_epidx_00', 'target_epidx_01']
    whitelist = np.setdiff1d(DF.columns,blacklist)
    dtypes    = DF[whitelist].dtypes
    outcols   = blacklist.copy()
    outtypes  = list(DF[blacklist].dtypes)

    # Ensure the dataframe is structured correctly
    DF = DF.sort_values(by=['uid','file','t_start']).reset_index(drop=True)

    # Find the indices for each unique file and user
    uid_file_inds      = DF.groupby(['uid','file']).indices
    uid_file_inds_keys = list(uid_file_inds.keys())

    # Rename the output columns
    attention_columns = []
    attention_types   = []
    for idx in range(window_size):
        attention_columns.extend([f'{val}_{idx:02d}' for val in whitelist])
        attention_types.extend(dtypes)
    attention_columns = np.array(attention_columns)
    attention_types   = np.array(attention_types)
    outcols.extend(attention_columns)
    outtypes.extend(attention_types)
    
    # Loop over the group index keys
    half_window = int((window_size-1)/2.)
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
                DF_slice        = DF.loc[irow[half_window]]
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

    # Make the output object
    outarr  = np.hstack((outref,outDF))
    outDF   = PD.DataFrame(outarr,columns=outcols)

    # Update the type
    for idx,icol in enumerate(outcols):
        outDF[icol] = outDF[icol].astype(outtypes[idx])
    
    # Save the result
    #outDF.to_pickle(outpath)
    return outDF,whitelist

def pivot_to_attention(argv):

    # Read in the data
    DF = PD.read_pickle(argv[1])
    make_windowed(DF,argv[2])

def pca_from_vector(DL_objects):

    # Make the output object
    out_train = PD.DataFrame()
    out_test  = PD.DataFrame()

    # Read in the channel names
    channels = yaml.safe_load(open('../../scripts/scalp_deep-learning_pnes-predictor/configs/channels.yaml','r'))

    # Get the tensor block information in a format we can work with
    block   = []
    colname = []
    for ikey,ival in DL_objects[0].items():
        for jval in ival:
            block.append(ikey)
            colname.append(jval)

    # Make a dataframe to manipulate for PCA
    DF            = PD.DataFrame({'block':block,'orig_col':colname})
    DF['channel'] = ''
    DF['feature'] = ''
    DF['n95']     = 1

    # Get the granularity needed for pca
    channel_list=DF['channel'].values
    feature_list=DF['feature'].values
    for idx,icol in enumerate(DF.orig_col.values):
        for ichannel in channels['channels']:
            if ichannel in icol:
                ifeature=icol.strip(f"{ichannel}_")
                channel_list[idx]=ichannel
                feature_list[idx]=ifeature
    DF['feature'] = feature_list
    DF['channel'] = channel_list

    # Get the PCA inds
    PCA_inds = DF.groupby(['block','feature']).indices

    # Loop through the inds, determine new feature names
    new_model_blocks = DL_objects[0].copy()
    new_model_blocks['frequency'] = []
    new_model_blocks['time']      = []
    for ikey in PCA_inds.keys():
        if ikey[0] in ['frequency','time']:
            
            # Get the data slice columns
            cols_to_fit = DF.loc[PCA_inds[ikey]]['orig_col'].values

            # get the data to pca fit
            traindata = DL_objects[1][cols_to_fit].values
            testdata  = DL_objects[2][cols_to_fit].values

            # Fit the data using a PCA
            PCA_object = PCA()
            PCA_object.fit(traindata)

            # Get the cumulative feature importance
            explained_variance = np.cumsum(PCA_object.explained_variance_ratio_)
            xvector            = np.arange(explained_variance.size)+1
            
            # Get the number of features closest to 95% of the covariance
            n95 = int(np.round(np.interp(0.95,explained_variance,xvector)))

            # Update the dataframe with info about the channels
            DF.loc[PCA_inds[ikey],['n95']] = n95

            # Fit using the 95 percentile component
            PCA_transformer = PCA(n_components=n95)
            PCA_transformer.fit(traindata)
            new_traindata = PCA_transformer.transform(traindata)
            new_testdata  = PCA_transformer.transform(testdata)
            
            # Make the new column names
            outcols = []
            for idx in range(n95):
                outcols.append(f"{ikey[1]}_pca{idx+1:02d}")
            
            # Make the temporary dataframe
            iDF = PD.DataFrame(new_traindata,columns=outcols)
            jDF = PD.DataFrame(new_testdata,columns=outcols)
            
            # Add this to the output dataframe
            out_train = PD.concat((out_train,iDF),axis=1)
            out_test  = PD.concat((out_test,jDF),axis=1)

            # Update the model block info
            new_model_blocks[ikey[0]].extend(outcols)
        else:

            # Get the columns to transpose directly
            cols_to_move = DF.loc[PCA_inds[ikey]]['orig_col'].values

            # grab the data slice
            traindata = DL_objects[1][cols_to_move]
            testdata  = DL_objects[2][cols_to_move]
            
            # Add this to the output dataframe
            out_train = PD.concat((out_train,traindata),axis=1)
            out_test  = PD.concat((out_test,testdata),axis=1)

    # Apply a downcast to 32 point precision. Will be important for making an attention vector
    for icol in out_train.columns:
        if out_train[icol].dtype == 'float64':
            out_train[icol] = out_train[icol].astype('float32')
            out_test[icol]  = out_test[icol].astype('float32')
        elif out_train[icol].dtype == 'int64':
            out_train[icol] = out_train[icol].astype('int32')
            out_test[icol]  = out_test[icol].astype('int32')

    return (new_model_blocks,out_train,out_test),DF

if __name__ == '__main__':

    #fpath             = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/MODELS/SSL/DATA/vector_data.pickle'
    #DL_objects        = pickle.load(open(fpath,'rb'))
    #new_DL_objects,DF = pca_from_vector(DL_objects)

    # get the attention network arrays
    fpath             = argv[1]
    DL_objects        = pickle.load(open(fpath,'rb'))
    traindf,whitelist = make_windowed(DL_objects[1])
    testdf,_          = make_windowed(DL_objects[2])

    # Make the new model block
    model_blocks = DL_objects[0]
    new_model_blocks = {}
    for ikey in model_blocks:
        new_model_blocks[ikey] = []
        for ival in model_blocks[ikey]:
            if ival in whitelist:
                newvals=[f'{ival}_{idx:02d}' for idx in range(9)]
                new_model_blocks[ikey].extend(newvals)
            else:
                new_model_blocks[ikey].append(ival)

    pickle.dump((new_model_blocks,traindf,testdf),open(argv[2],'wb'))
