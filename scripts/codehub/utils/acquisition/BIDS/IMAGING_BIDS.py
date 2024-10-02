import glob
import json
import shutil
import pickle
import argparse
import numpy as np
from pathlib import Path as Pathlib

# Pybids imports
from bids import BIDSLayout
from bids.layout.writing import build_path

class prepare_imaging:

    def __init__(self,args):
        self.args          = args
        self.newflag       = False

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
            
            # Get the bids keys
            bidskeys = self.get_protocol(ifile)
            
            # Save the results
            print(ifile, bidskeys)
            self.save_data(ifile,bidskeys)

        # Update data lake as needed
        self.update_datalake()

    def update_datalake(self):
        # Ask if the user wants to save the updated datalake
        if self.newflag:
            flag = input("Save the new datalake entires (Yy/Nn)? ")
            if flag.lower() == 'y':
                newpath = input("Provide a new filename: ")
            outlake = {'HUP':self.datalake}
            pickle.dump(outlake,open(newpath,'wb'))

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
                    newval = None
                output[ikey] = newval
        
            # Update the datalake
            self.datalake[series] = output
            self.keys = np.array(list(self.datalake.keys()))

        return output

    def save_data(self,ifile,bidskeys):

        # Update keywords
        entities  = {}

        # Define the required keys
        entities['subject']     = self.args.subject
        entities['session']     = self.args.session
        entities['run']         = self.args.run
        entities['datatype']    = bidskeys['data_type']

        # Begin building the match string
        match_str = 'sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}]'

        # Optional keys
        if type(bidskeys['task']) == str or not np.isnan(bidskeys['task']):
            entities['task']        = bidskeys['task']
            match_str += '[_task-{task}]'
        if type(bidskeys['acq']) == str or not np.isnan(bidskeys['acq']):
            entities['acquisition'] = bidskeys['acq']
            match_str += '[_acq-{acquisition}]'
        if type(bidskeys['ce']) == str or not np.isnan(bidskeys['ce']):
            entities['ceagent'] = bidskeys['ce']
            match_str += '[_ce-{ceagent}]'

        # Add in the run number here
        match_str += '[_run-{run}]'

        # Remaining optional keys
        if type(bidskeys['modality']) == str or not np.isnan(bidskeys['modality']):
            entities['modality'] = bidskeys['modality']
            match_str += '[_{modality}]'

        # Define the patterns for pathing    
        patterns = [match_str]

        # Set up the bids pathing
        bids_path = self.args.bidsroot+build_path(entities=entities, path_patterns=patterns)

        # Save the nifti to its new home
        rootpath = '/'.join(bids_path.split('/')[:-1])
        Pathlib(rootpath).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ifile, bids_path)

        # Create a new BIDSLayout object
        layout = BIDSLayout(args.bidsroot)

        # Save the bids layout
        output_path = os.path.join(args.bidsroot, 'dataset_description.json')
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
        json_output = layout.to_df().to_dict()
        merged_data = {**existing_data, **json_output}
    
        # Save the updated data back to the JSON file
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=4)

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