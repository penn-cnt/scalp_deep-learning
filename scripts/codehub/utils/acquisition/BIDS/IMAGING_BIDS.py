import glob
import json
import pickle
import argparse
import numpy as np
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
                    newval = np.nan
                output[ikey] = newval
        
            # Update the datalake
            self.datalake[series] = output
            self.keys = np.array(list(self.datalake.keys()))

        return output

    def save_data(self,ifile,bidskeys):

        # Update keywords
        entities  = {}
        # Required keys
        entities['subject']     = self.args.subject
        entities['session']     = self.args.session
        entities['run']         = self.args.run
        entities['datatype']    = bidskeys['data_type']

        # Optional keys
        if bidskeys['modality'] != None:
            entities['modality']    = bidskeys['modality']
        if bidskeys['task'] != None:
            entities['task']        = bidskeys['task']
        if bidskeys['acq'] != None:
            entities['acquisition'] = bidskeys['acq']
        if bidskeys['ceagent'] != None:
            entities['ceagent']     = bidskeys['ce']

        # Define the patterns for pathing    
        patterns = ['sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}][_acq-{acquisition}][_ce-{ceagent}][_run-{run}][_{modality}].{extension<nii|nii.gz|json|bval|bvec|json>|nii.gz}']

        # Set up the bids pathing
        bids_path = self.args.bidsroot+build_path(entities=entities, path_patterns=patterns)
        print(bids_path)

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