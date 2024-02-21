import os
import pickle
import argparse
import subprocess
import numpy as np
import pandas as PD
from sys import exit
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

class audit:

    def __init__(self,search_root,outdir,ostype,cmd_path,audit_history,username):

        # Save the inputs to class instance
        self.rootdir       = search_root
        self.outdir        = outdir
        self.os            = ostype
        self.cmd_path      = cmd_path

        if not os.path.exists(self.outdir):
            os.system(f"mkdir -p {self.outdir}")

        # Define the delimter we will use for folder breaks
        self.delimiter     = '.'

        # Some basic cleanup to ensure we have trailing directory slashes
        if self.rootdir[-1] != '/':
            self.rootdir = self.rootdir+'/'
        if self.outdir[-1] != '/':
            self.outdir = self.outdir+'/'

        # Get the audit history
        if audit_history != None:
            self.audit_history = audit_history
        else:
            self.audit_history = self.outdir+'audit_history_{username}.csv'

        # Create a temporary file that stores all of the input paths for the given root directory. This speeds up runs if testing/restarting.
        fname           = f"{self.rootdir.replace('/',self.delimiter)}inputs"
        if fname[0] == self.delimiter:
            fname = fname[1:]
        self.input_file = f"{self.outdir}{fname}"

        # Create a lock file. This is meant to prevent parallel processes from opening a file at the same time
        self.lock_file  = self.outdir+'audit_history_{username}.lock'

        # Output audit location
        self.audit_data = self.outdir+'audit_data_{username}.csv'

    def argcheck(self):
        """
        Make sure the user provided the correct inputs to run the script. Looks for root directory to search through, and a correct filesystem type.
        """

        # Read in the user provided root directory and make sure it exists
        if not os.path.exists(self.rootdir):
            raise NameError("No directory found matching that name.")
                
        # Make sure the os archiecture is allowed
        if self.os.lower() not in ['unix','windows']:
            raise NameError("Please specify 'unix' or 'windows' back-end.")
        
    def read_cmd(self):
        """
        Read in the filesystem specific command needed to run the audit.
        """

        # Read in the relevant config
        fp              = open(self.cmd_path)
        self.cmd_master = fp.read()
        fp.close()

    def define_inputs(self):

        # Locally scoped function to travese directories to get finer granularity input list
        def get_all_subdirectories(directory):
            """Recursively get all subdirectories."""
            subdirectories = []
            for entry in os.scandir(directory):
                if entry.is_dir():
                    subdirectories.append(entry.path)
                    subdirectories.extend(get_all_subdirectories(entry.path))
            return subdirectories

        # Define the directories in the root directory
        if os.path.exists(self.input_file):
            self.folders = pickle.load(open(self.input_file,'rb'))
        else:
            folders = get_all_subdirectories(self.rootdir)
            self.folders = np.sort(folders)
            pickle.dump(self.folders,open(self.input_file,"wb"))

        # Read in the audit history
        if os.path.exists(self.audit_history):
            self.history = PD.read_csv(self.audit_history)
        else:
            self.history = PD.DataFrame(columns=['directory_path','mjd'])

        # Check for already completed entries
        self.input_paths  = []
        completed_folders = self.history['directory_path'].values
        for ifolder in self.folders:
            alt_path = str(ifolder) + '/'
            if ifolder not in completed_folders and alt_path not in completed_folders:
                self.input_paths.append(str(ifolder))
            else:
                print(f"Skipping {ifolder}.")
        self.input_paths.append(str(self.rootdir))

        # Clean up the paths as needed
        for idx in range(len(self.input_paths)):
            if self.input_paths[idx][-1] != '/':
                self.input_paths[idx] += '/'

    def perform_audit_linux(self):
        """
        Perform a data audit on a linux/unix filesystem.
        """

        for idx,ifolder in tqdm(enumerate(self.input_paths), desc='Audit: ', total=len(self.input_paths)):

            # User update
            #print(f"Performing audit on {ifolder}.")

            # Save the input string to a different name in case of modifications
            instr = ifolder

            # Modify the input directory name to make an output filename
            self.outname   = f"{ifolder.replace('/',self.delimiter)[:-1]}.audit"
            if self.outname[0] == self.delimiter:
                self.outname = self.outname[1:]
            self.outname = f"{self.outdir}{self.outname}"

            # Update the cmd string for this case
            cmd = self.cmd_master.replace("INDIR_SUBSTR",instr)
            cmd = cmd.replace("OUTDIR_SUBSTR",self.outname)
            
            # Run command
            subprocess.run(cmd, shell=True, check=True)

            # Update audit history
            self.history.loc[len(self.history.index)] = [ifolder,datetime.now().timestamp()]

            # Check if lock file is active
            while os.path.exists(self.lock_file):
                time.sleep(1)

            # Write the lock file so another processing cant write to the audit history yet
            with open(self.lock_file, "w") as lock_file:
                lock_file.write("locked")

            # Write the history and remove the lock
            self.history.to_csv(self.audit_history,index=False)
            os.remove(self.lock_file)

    def clean_audit(self):
        """
        Clean up the audit outputs to one csv file.
        """

        # Make or load the audit data file as needed
        if not os.path.exists(self.audit_data):
            fp = open(self.audit_data,'w')
            fp.write('path,md5,size-(MB),last-modified-date,owner\n')
        else:
            fp = open(self.audit_data,'a')

        # Add the root directory to the search path, then loop
        folder_list = [str(ifolder) for ifolder in self.folders]
        folder_list.append(str(self.rootdir))
        for ifolder in folder_list:

            # Get the expected target output file for this folder
            self.outname   = f"{ifolder.replace('/',self.delimiter)}.audit"
            if self.outname[0] == self.delimiter:
                self.outname = self.outname[1:]
            self.outname = f"{self.outdir}{self.outname}"
            self.outname = self.outname.replace('..audit','.audit')

            # If file exists, add it to the data file then remove
            if os.path.exists(self.outname):
                
                # Read in and clean the raw audit data
                fp2     = open(self.outname,'r')
                rawdata = np.array(fp2.readlines())
                rawdata = rawdata.reshape((-1,4))
                for irow in rawdata:
                    rawstring  = ' '.join(irow)
                    clean_data = rawstring.replace('\n','').replace(' MB','').replace(' ',',')
                    fp.write(f"{clean_data}\n")
                    rawdata = fp2.readline()
                fp2.close()
                os.remove(self.outname)
        fp.close()

if __name__ == '__main__':
    
    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--search_root", type=str, required=True, help="Root directory to recursively audit down from.")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory to store the audit.")
    parser.add_argument("--os", type=str, default='unix', help="OS architecture. Allowed Arguments='unix' or 'windows'.")
    parser.add_argument("--cmd_path", type=str, default='config/audit.md5sum.linux', help="Path to command string to execute.")
    parser.add_argument("--audit_history", type=str, help="Path to the audit history.")
    parser.add_argument("--merge", action='store_true', default=False, help="Merge outputs to final audit file.")
    parser.add_argument("--username", type=str, default='main', help="Username for data audit.")
    args = parser.parse_args()

    # Run through the audit
    AH = audit(args.search_root,args.outdir,args.os,args.cmd_path,args.audit_history,args.username)
    AH.argcheck()
    AH.read_cmd()
    AH.define_inputs()
    if args.os.lower() == 'unix':
        AH.perform_audit_linux()
    if args.merge:
        AH.clean_audit()
    