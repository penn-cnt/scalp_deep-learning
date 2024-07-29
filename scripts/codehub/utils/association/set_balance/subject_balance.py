import argparse
import numpy as np
import pandas as PD

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--infile", type=str, help='Input Filepath')
    parser.add_argument("--outfile", type=str, help='Output Filepath')
    parser.add_argument("--nsubject", type=int, default=50, help='Block size of files to try and ensure balance within')
    parser.add_argument("--blocksize", type=int, default=500, help='Block size of files to try and ensure balance within')
    args = parser.parse_args()

    # Read in the file
    DF = PD.read_csv(args.infile)

    # Make subject number column
    DF['subnum']=DF['filepath'].apply(lambda x:int(x.split('sub-')[-1].split('_ses')[0]))

    # Get the unique subject ids that fall within the threshold
    usubs = np.sort(DF['subnum'].unique())[:args.nsubject]

    # Make the split on subject number
    DF_within = DF.loc[DF.subnum.isin(usubs)].reset_index(drop=True)
    DF_remain = DF.loc[~DF.subnum.isin(usubs)].reset_index(drop=True)

    # Loop over the block size and grab from each subject one by one
    npersub = int(args.blocksize / args.nsubject)

    # Loop over the subjects, grab the first number of files to properly fill the blocksize, store index so we can repopulate properly
    header_inds = np.array([])
    for isub in usubs:
        iDF         = DF_within.loc[DF_within.subnum == isub]
        header_inds = np.concatenate((header_inds,list(iDF.index[:npersub])))
    header_inds = header_inds.astype('int16')
    remain_inds = np.setdiff1d(DF_within.index,header_inds)

    # Create the new organized dataframe
    outDF = PD.concat((DF_within.loc[header_inds],DF_within.loc[remain_inds]))
    outDF = PD.concat((outDF,DF_remain))
    outDF = outDF.drop(['subnum'],axis=1)
    outDF.to_csv(args.outfile,index=False)
