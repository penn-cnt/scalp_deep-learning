import os
import glob
import argparse
import numpy as np
import pandas as PD
from mne.io import read_raw_persyst
from mne_bids import BIDSPath, write_raw_bids

def DateException(inpath):

    # Read in the lay file
    DF = PD.read_csv(inpath,delimiter='=',names=['key','value'])
    
    # Grab the original value
    global original_testtime
    original_testime = DF.loc[DF.key=='TestTime']['value'].values[0]

    # Clean up the times
    new_testtime = original_testime.split('.')[:-1][0]

    # Make a backup of the lay file
    os.system(f"cp {inpath} {inpath}.lock")

    # Replace the bad formatting
    os.system(f"sed -i '' 's/{original_testime}/{new_testtime}/g' {inpath}")

def cleanup(inpath):
    # Restore the original lay file
    os.system(f"mv {inpath}.lock {inpath}")

def read_lay_data(inpath):
    """
    Try to read in the lay data. If we encounter a floating point error, try a simple fix. If it fails, stop there.

    Args:
        inpath (str): Path to .lay file.
    """

    # Test if the data can be read in
    try:
        raw = read_raw_persyst(inpath)
    except ValueError:
        DateException(inpath)
        raw = read_raw_persyst(inpath)
    return raw

def save_bids(raw,bidsroot,subject_num,session_num):

        # Make the BIDS path
        bids_path = BIDSPath(root=bidsroot, datatype='eeg', session=f"implant{session_num:03d}", subject=f"{subject_num:03d}", run=1, task='task')

        # Save the bids data
        write_raw_bids(bids_path=bids_path, raw=raw, allow_preload=True, format='EDF',verbose=False)

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--laypath", type=str, required=True, help='Path to .lay file. MNE assumes the associated .dat is in the same folder.')
    parser.add_argument("--edfpath", type=str, required=True, help='Path to root BIDS directory for .edf file.')
    parser.add_argument("--wildcard", action='store_true', default=False, help="Laypath is a wildcard path.")
    parser.add_argument("--outpath", type=str, help='Path to output summary stats.')
    args = parser.parse_args()

    # Assign filepathing
    if args.wildcard == False:
        laypaths = [args.laypath]
    else:
        laypaths = glob.glob(args.laypath)
        laypaths = np.sort(laypaths)

    # Get Neuroimaging mappings
    SESSION_MAPPING = {}

    # Make the dataset
    summary_df = PD.DataFrame(columns=['file','subject','session','date','nchan','nsamp','fs'])
    for ifile in laypaths:

        # Get the HUP number
        HUP_NUM = int(ifile.split("HUP")[-1].split("_")[0])

        # Get the session_number
        if HUP_NUM not in SESSION_MAPPING.keys():
            SESSION_MAPPING[HUP_NUM] = 1
        else:
            SESSION_MAPPING[HUP_NUM] += 1
        
        # Apply mapping to neuroimaging
        SUB_NUM = HUP_NUM
        SES_NUM = SESSION_MAPPING[HUP_NUM]

        try:
            # Get the lay data into memory
            raw = read_lay_data(ifile)

            # Add info to the summary df
            idata = raw.get_data()
            iarr  = np.array([ifile.split('/')[-1],SUB_NUM,SES_NUM,raw.info["meas_date"],idata.shape[0],idata.shape[1],raw.info["sfreq"]]).reshape((1,-1))
            iDF = PD.DataFrame(iarr,columns=['file','subject','session','date','nchan','nsamp','fs'])
            summary_df = PD.concat((summary_df,iDF))

            # Some cleanup and anonymization steps
            raw.info['subject_info']['birthday'] = None
            raw.info['subject_info']['sex']      = None
            raw.anonymize()

            # Save the output
            save_bids(raw,args.edfpath,SUB_NUM,SES_NUM)
        except Exception as e:
            print(f"Encountered error {e}")
            iarr  = np.array([ifile.split('/')[-1],SUB_NUM,SES_NUM,'','','','']).reshape((1,-1))
            iDF = PD.DataFrame(iarr,columns=['file','subject','session','date','nchan','nsamp','fs'])
            summary_df = PD.concat((summary_df,iDF))

        # Cleanup the results
        cleanup(ifile)

        # Save the summary
        if args.outpath != None:
            summary_df.to_csv(args.outpath,index=False)
