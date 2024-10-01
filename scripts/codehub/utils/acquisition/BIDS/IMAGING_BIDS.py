import glob
import json
import pickle
import argparse
import numpy as np

# Local import
from components.internal.BIDS_handler import *
from components.internal.observer_handler import *

class prepare_imaging(Subject):

    def __init__(self,args):
        self.args          = args
        self.newflag       = False
        self.BIDS_keywords = {'root':self.args.bidsroot,'datatype':None,'session':None,'subject':None,'run':None,'task':None}

        # Create the object pointers
        self.BH = BIDS_handler()

    def workflow(self):
        """
        Workflow to turn a flat folder of imaging data to BIDS
        """

        # Attach observers
        self.attach_objects()

        # get json paths
        self.get_filepaths()
        
        # Load the datalake
        self.load_datalake()

        # Loop over the files
        for ifile in self.json_files:
            bidskeys = self.get_protocol(ifile)
            self.save_data(ifile,bidskeys)

        # Ask if the user wants to save the updated datalake
        if self.newflag:
            flag = input("Save the new datalake entires (Yy/Nn)? ")
            if flag.lower() == 'y':
                newpath = input("Provide a new filename: ")
            outlake = {'HUP':self.datalake}
            pickle.dump(outlake,open(newpath,'wb'))

    def attach_objects(self):
        """
        Attach observers here so we can have each multiprocessor see the pointers correctly.
        """

        # Create the observer objects
        self._meta_observers = []

        # Attach observers
        self.add_meta_observer(BIDS_observer)

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
            
            # Alert code that we updated the datalake
            self.newflag = True

            # Get new inputs
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

        return output

    def save_data(self,ifile,bidskeys):
        # Update keywords
        self.keywords = {'filename':ifile,'root':self.args.bidsroot,'datatype':bidskeys['data_type'],
                            'session':self.args.session,'subject':self.args.subject,'run':self.args.run,
                            'task':bidskeys['task'],'fs':None,'start':0,'duration':0,'uid':0}
        self.notify_metadata_observers()

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser   = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Input path to the folder containing niftii files.')
    parser.add_argument('--bidsroot', required=True, help='Output path to the BIDS root directory.')
    parser.add_argument('--datalake', help='Output path to the bids datalake for image naming.',default="./datalakes/HUP_BIDS_DATALAKE.pickle")
    parser.add_argument('--subject', required=True, help='Subject label.')
    parser.add_argument('--session', required=True, help='Session label.')
    parser.add_argument('--run', required=True, help='Run label.')
    args = parser.parse_args()

    # Minor cleanuo
    if args.dataset[-1] != '/':args.dataset += '/'

    # Prepare data for BIDS work
    PI = prepare_imaging(args)
    PI.workflow()