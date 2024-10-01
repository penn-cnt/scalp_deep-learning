import glob
import json
import pickle
import argparse
import numpy as np

class prepare_imaging:

    def __init__(self,args):
        self.args = args

    def workflow(self):
        """
        Workflow to turn a flat folder of imaging data to BIDS
        """

        # get json paths
        self.get_filepaths()
        
        # Load the datalake
        self.load_datalake()

        # Loop over the files
        for ifile in self.json_files:
            self.get_protocol(ifile)

    def get_filepaths(self):
        self.json_files = glob.glob(f"{self.args.dataset}*json")

    def load_datalake(self):
        self.datalake = pickle.load(open(self.args.datalake,'rb'))['HUP']
        self.keys     = np.array(list(self.datalake.keys()))

    def get_protocol(self,infile):

        # Open the metadata
        metadata = json.load(open(infile,'r'))
        
        # get the protocol name
        series     = metadata["ProtocolName"].lower()
        series_alt = series.replace(' ','_')

        # get the appropriate keywords
        if series in self.keys:
            output = self.datalake[series]
        elif series_alt in self.keys:
            output = self.datalake[series_alt]
        else:
            output = {}
        
        # If we are missing information, ask the user
        keys_to_request = ['scan_type','data_type', 'modality', 'task', 'acq', 'ce']
        if not output.keys():
            print(f"Please provide information for {series}")
            for ikey in keys_to_request:
                if ikey == 'data_type':
                    while True:
                        newval = input("Data Type (Required): ")
                        if newval != '':
                            break
                else:
                    newval = input(f"{ikey} (''=None): ")
                if newval == '':
                    newval = np.nan
                output[ikey] = newval
        
            # Update the datalake
            self.datalake[series] = output
            self.keys = np.array(list(self.datalake.keys()))

        # print results for testing
        print(series,output)
        


if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser   = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Input path to the folder containing niftii files.')
    parser.add_argument('--bidsroot', required=True, help='Output path to the BIDS root directory.')
    parser.add_argument('--datalake', help='Output path to the bids datalake for image naming.',default="./datalakes/HUP_BIDS_DATALAKE.pickle")
    args = parser.parse_args()

    # Minor cleanuo
    if args.dataset[-1] != '/':args.dataset += '/'

    # Prepare data for BIDS work
    PI = prepare_imaging(args)
    PI.workflow()