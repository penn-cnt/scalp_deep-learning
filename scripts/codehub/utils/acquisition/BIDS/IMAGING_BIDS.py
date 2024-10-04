import re
import os
import glob
import json
import shutil
import pickle
import argparse
import numpy as np
from sys import exit
from tqdm import tqdm
from pathlib import Path as Pathlib
from prettytable import PrettyTable,ALL

# Pybids imports
from bids import BIDSLayout
from bids.layout.writing import build_path

class prepare_imaging:

    def __init__(self,args):
        self.args     = args
        self.newflag  = False
        self.lakekeys = ['data_type', 'scan_type', 'modality', 'task', 'acq', 'ce']

    def workflow(self):
        """
        Workflow to turn a flat folder of imaging data to BIDS
        """

        # get json paths
        self.get_filepaths()
        
        # Load the datalake
        self.load_datalake()

        # Make the dataset description
        self.make_description()

        # Infer session labels as needed
        if self.args.session == None:
            self.infer_sessions()

        # Loop over the files
        for ifile in tqdm(self.json_files, total=len(self.json_files), desc="Making BIDS"):
            
            # Get the bids keys
            bidskeys = self.get_protocol(ifile)
            
            # Save the results
            self.save_data(ifile,bidskeys)

        # Update data lake as needed
        self.update_datalake()

    def infer_sessions(self):

        # Loop over the files and get the unique sessions
        pattern = r'_(\d{14})(_|\.?)'
        dates   = []
        for ifile in self.json_files:
            # Get the relevant substring
            match  = re.search(pattern, ifile)
            substr = match.group(1)[:8]
            dates.append(int(substr))
        
        # Get the unique sorted options
        udates = np.unique(dates)
        
        # Get the session mapping
        self.session_map = {}
        for idx,ifile in enumerate(self.json_files):
            self.session_map[ifile] = np.where(udates==dates[idx])[0][0]

    def make_description(self):

        dataset_description = {
            'Name': 'Your Dataset Name',
            'BIDSVersion': '1.6.0',
            'Description': 'Description of your dataset',
            'License': 'License information'
            }

        # Save the dataset description as JSON
        with open(f"{self.args.bidsroot}dataset_description.json", 'w') as f:
            json.dump(dataset_description, f, indent=4)

    def update_datalake(self):
        # Ask if the user wants to save the updated datalake
        if self.newflag:
            flag = input("Save the new datalake entires (Yy/Nn)? ")
            if flag.lower() == 'y':
                newpath = input("Provide a new filename: ")
                outlake = {'HUP':self.datalake}
                pickle.dump(outlake,open(newpath,'wb'))

    def get_filepaths(self):
        # Find all the json files in the flat data folder
        self.json_files = glob.glob(f"{self.args.dataset}*json")

    def load_datalake(self):
        # Open the datalake and store the protocol name keys to selkf
        self.datalake = pickle.load(open(self.args.datalake,'rb'))['HUP']
        self.keys     = np.array(list(self.datalake.keys()))

    def acquire_keys(self,iprotocol):
        """
        Acquire keys from the user for the current protocol name
        """

        # Alert code that we updated the datalake
        self.newflag = True

        # Make the output object and query keys
        output = {}

        # Get new inputs
        print(f"Please provide information for {iprotocol}")
        for ikey in self.lakekeys:
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
        self.datalake[iprotocol] = output
        self.keys = np.array(list(self.datalake.keys()))

    def print_protocol(idict):
        
        # Make a pretty table to make interpreting results easier
        table = PrettyTable(hrules=ALL)
        tcols = [series]
        tcols.extend(self.lakekeys)
        table.field_names = tcols
        row = [series]
        for ikey in self.lakekeys:
            row.append(idict[ikey])
        table.add_row(row)
        table.align['path'] = 'l'

        # get user input if this is okay
        while true:
            user_input = input("Is this okay (Yy/Nn)?")
            if user_input.lower() in ['y','n']:
                break
        return user_input

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

        # Get/confirm information
        if not output.keys():
            self.acquire_keys(series)
        else:
            passflag = self.print_protocol(output)
            print(passflag)

        return output

    def save_data(self,ifile,bidskeys):

        # Update keywords
        entities  = {}

        # Define the required keys
        entities['subject']     = self.args.subject
        entities['run']         = self.args.run
        
        # Check for undefined data type
        datatype = bidskeys['data_type']
        if type(datatype) != str:
            print(ifile)
            exit()
        entities['datatype'] = bidskeys['data_type']



        # Get the session label
        if self.args.session != None:
            entities['session'] = self.args.session
        else:
            entities['session'] = f"preprocessor{self.session_map[ifile]:02d}"

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

        # Make the folder to save to
        rootpath = '/'.join(bids_path.split('/')[:-1])
        Pathlib(rootpath).mkdir(parents=True, exist_ok=True)
        
        # Copy the different data files over
        root_file     = '.'.join(ifile.split('.')[:-1])
        current_files = glob.glob(f"{root_file}*")
        for jfile in current_files:
            extension = jfile.split('.')[-1]  
            shutil.copyfile(jfile, f"{bids_path}.{extension}")

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
    parser.add_argument('--session', help='Session label. If blank, try to infer from filename.')
    parser.add_argument('--run', default=1, help='Run label.')
    args = parser.parse_args()

    # Minor cleanuo
    if args.dataset[-1] != '/':args.dataset += '/'
    if args.bidsroot[-1] != '/':args.bidsroot += '/'

    # Prepare data for BIDS work
    PI = prepare_imaging(args)
    PI.workflow()