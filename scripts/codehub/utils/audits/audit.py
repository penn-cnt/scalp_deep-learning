import os
import argparse
import subprocess
import numpy as np
import pandas as PD
from sys import exit
from pathlib import Path
from datetime import datetime

class audit:

    def __init__(self,search_root,outdir,os,cmd_path,audit_history):
        self.rootdir       = search_root
        self.outdir        = outdir
        self.os            = os
        self.cmd_path      = cmd_path

        # Some basic cleanup
        if self.rootdir[-1] != '/':
            self.rootdir = self.rootdir+'/'
        if self.outdir[-1] != '/':
            self.outdir = self.outdir+'/'

        # Get the audit history
        if audit_history != None:
            self.audit_history = audit_history
        else:
            self.audit_history = self.outdir+'audit_history.csv'

    def argcheck(self):

        # Read in the user provided root directory and make sure it exists
        if not os.path.exists(self.rootdir):
            raise NameError("No directory found matching that name.")
                
        # Make sure the os archiecture is allowed
        if self.os.lower() not in ['unix','windows']:
            raise NameError("Please specify 'unix' or 'windows' back-end.")
        
    def read_cmd(self):

        # Read in the relevant config
        fp              = open(self.cmd_path)
        self.cmd_master = fp.read()
        fp.close()

    def define_inputs(self):

        # Define the directories in the root directory
        folders       = []
        root_contents = Path(self.rootdir).glob("*")
        for content in root_contents:
            if os.path.isdir(content):
                folders.append(content)
        folders = np.sort(folders)

        # Read in the audit history
        if os.path.exists(self.audit_history):
            self.history = PD.read_csv(self.audit_history)
        else:
            self.history = PD.DataFrame(columns=['directory_path','mjd'])

        # Check for already completed entries
        self.input_paths  = []
        completed_folders = self.history['directory_path'].values
        for ifolder in folders:
            if ifolder not in completed_folders:
                self.input_paths.append(str(ifolder))
        self.input_paths.append(str(self.rootdir))

        # Clean up the paths as needed
        for idx in range(len(self.input_paths)):
            if self.input_paths[idx][-1] != '/':
                self.input_paths[idx] += '/'

    def perform_audit_linux(self):

        for ifolder in self.input_paths:

            # Modify the input directory name to make an output filename
            delimiter = '.'
            self.outname   = f"{ifolder.replace('/',delimiter)[:-1]}.audit"
            if self.outname[0] == delimiter:
                self.outname = self.outname[1:]
            self.outname = f"{self.outdir}{self.outname}"

            # Update the cmd string for this case
            cmd = self.cmd_master.replace("INDIR_SUBSTR",ifolder)
            cmd = cmd.replace("OUTDIR_SUBSTR",self.outname)
            
            # Try Except catch our shell command
            try:
                subprocess.run(cmd, shell=True, check=True)
                self.history.loc[len(self.history.index)] = [ifolder,datetime.now().timestamp()]
                self.history.to_csv(self.audit_history)
            except:
                pass

if __name__ == '__main__':
    
    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--search_root", type=str, required=True, help="Root directory to recursively audit down from.")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory to store the audit.")
    parser.add_argument("--os", type=str, default='unix', help="OS architecture. Allowed Arguments='unix' or 'windows'.")
    parser.add_argument("--cmd_path", type=str, default='config/audit.md5.linux', help="Path to command string to execute.")
    parser.add_argument("--audit_history", type=str, help="Path to the audit history.")
    args = parser.parse_args()

    # Run through the audit
    AH = audit(args.search_root,args.outdir,args.os,args.cmd_path,args.audit_history)
    AH.argcheck()
    AH.read_cmd()
    AH.define_inputs()
    if args.os.lower() == 'unix':
        AH.perform_audit_linux()

    