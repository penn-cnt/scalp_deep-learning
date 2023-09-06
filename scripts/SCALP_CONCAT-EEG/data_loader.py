# Libraries to help path complete raw inputs
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# Torch imports
import torch

# General libraries
import re
import glob
import time
import resource
import argparse
import numpy as np
import pandas as PD
from sys import exit
from  pyedflib import highlevel

class data_loader:
    """
    Class devoted to loading in raw data into the shared class instance.
    """

    def load_edf(self):
        """
        Parent class data loader for EDF file format.
        """

        # Load current edf data into memory
        self.raw_data, self.channel_metadata, self.scan_metadata = highlevel.read_edf(self.infile)
        header           = highlevel.read_edf_header(self.infile)
        self.channels    = header['channels']
        self.annotations = header['annotations']
        #for ikey in list(header.keys()):
        #    print(ikey)
        #    print(header[ikey])
        #    print("===")
        print(self.annotations)
        #exit()

        # Clean up the edf data
        self.channels = [ichannel.upper() for ichannel in self.channels]

        # Get the underlying data shapes
        self.ncol = len(self.raw_data)
        self.nrow = max([ival.size for ival in self.raw_data])
        
class channel_clean:
    """
    Class devoted to cleaning different channel naming conventions.
    """

    def __init__(self,clean_method='HUP'):
        if clean_method == 'HUP':
            self.HUP_clean()

    def HUP_clean(self):
        """
        Return the channel names according to HUP standards.
        Adapted from Akash Pattnaik code.
        Updated to handle labels not typically generated at HUP (All leters, no numbers.)
        """

        self.clean_channel_map = []
        for ichannel in self.channels:
            regex_match = re.match(r"(\D+)(\d+)", ichannel)
            if regex_match != None:
                lead        = regex_match.group(1).replace("EEG", "").strip()
                contact     = int(regex_match.group(2))
                new_name    = f"{lead}{contact:02d}"
            else:
                new_name = ichannel.replace("EEG","").replace("-REF","").strip()
            self.clean_channel_map.append(new_name)

class channel_mapping:
    """
    Class devoted to the different channel mappings of interest. 
    """

    def __init__(self,channel_mapping):
        if channel_mapping == "HUP1020":
            self.mapping_HUP_1020()

    def mapping_HUP_1020(self):
        """
        Mapping used to eventually build a 1020 model using HUP data.
        """

        self.master_channel_list = ['C03', 'C04', 'CZ', 'F03', 'F04', 'F07', 'F08', 'FZ', 'FP01', 'FP02', 'O01',
                                    'O02', 'P03', 'P04', 'T03', 'T04', 'T05', 'T06']
        self.channel_map_out = np.intersect1d(self.clean_channel_map,self.master_channel_list)

class channel_montage:
    """
    Class devoted to calculating the montages of interest for the data.
    """

    def __init__(self):
        if self.args.montage == "HUP1020":
            self.montage_HUP_1020()

    def montage_common_average(self):
        """
        Calculate the common average montage.
        """

        # Get the averages across all channels per time slice
        averages = np.average(self.dataframe.values(axis=0))
        averages = np.tile(averages, (1, self.ncol))

        # Get the montage data
        montage_data = self.dataframe.values-averages

        # Get the montage labels
        montage_channels = [f"{ichannel}-com_avg" for ichannel in self.dataframe.columns]

        # Make the montage dataframe
        dataframe_manager.montaged_dataframe(self,montage_data,montage_channels)

    def montage_HUP_1020(self):
        """
        Calculate the HUP 1020 montage and create the channel labels. Passes its data directly to the dataframe class.
        """

        # Channel structure for the montage
        bipolar_array = [['FP01','F07'],
                         ['F07','T03'],
                         ['T03','T05'],
                         ['T05','O01'],
                         ['FP02','F08'],
                         ['F08','T04'],
                         ['T04','T06'],
                         ['T06','O02'],
                         ['FP01','F03'],
                         ['F03','C03'],
                         ['C03','P03'],
                         ['P03','O01'],
                         ['FP02','F04'],
                         ['F04','C04'],
                         ['C04','P04'],
                         ['P04','O02'],
                         ['FZ','CZ']]
        
        # Get the new values to pass to the dataframe class
        montage_data = []
        for ival in bipolar_array:
            try:
                montage_data.append(list(self.dataframe[ival[0]].values-self.dataframe[ival[1]].values))
            except KeyError:
                montage_data.append(np.nan*np.ones(self.dataframe.shape[0]))
        montage_data = np.array(montage_data).T

        # Get the new montage channel labels
        self.montage_channels = [f"{ichannel[0]}-{ichannel[1]}" for ichannel in bipolar_array] 

        # Pass the data to the dataframe class function for montages
        dataframe_manager.montaged_dataframe(self,montage_data,self.montage_channels)

class dataframe_manager:
    """
    Class devoted to all things dataframe related. These dataframes will be used to populate the PyTorch tensors.
    """

    def __init__(self):

        self.dataframe = PD.DataFrame(index=range(self.nrow), columns=self.master_channel_list)
        values         = self.dataframe.values
        for idx,icol in enumerate(self.clean_channel_map):
            ivals      = self.raw_data[idx]
            try:
                column_idx                     = np.argwhere(self.dataframe.columns==icol)[0][0]
                values[:ivals.size,column_idx] = ivals
            except IndexError:
                pass

    def column_subsection(self,keep_columns):
        """
        Return a dataframe with only the columns requested.

        Args:
            keep_columns (list of channel labels): List of columns to keep.
        """

        # Get the columns to drop
        drop_cols = np.setdiff1d(self.dataframe.columns,keep_columns)
        self.dataframe = self.dataframe.drop(drop_cols, axis=1)

    def montaged_dataframe(self,data,columns):
        """
        Create a dataframe that stores the montaged data.

        Args:
            data (array): array of montaged data
            columns (list): List of column names
        """

        self.montaged_dataframe = PD.DataFrame(data,columns=columns)

class data_viability:

    def __init__(self):

        # Interpolate over small subsets of NaN if requested
        if self.args.interp:
            for idx,data_array in self.input_tensor_list:
                self.input_tensor_list[idx] = self.interpolate_data(data_array)

        # Find minimum viable datasets
        if self.args.viability == "VIABLE_DATA":

            # Loop over each dataset and find the ones that have no NaNs
            flags = []
            for idx,data_array in enumerate(self.input_tensor_list):
                flags.append(self.viable_data(data_array))
            
            # Output list
            output = []
            for idx,iarr in enumerate(self.input_tensor_list):
                if flags[idx]:
                    output.append(iarr)
            return output
        
        elif self.args.viability == 'VIABLE_COLUMNS':
            
            # Loop over each array and get the valid columns as boolean flag
            flags = []
            for idx,data_array in self.input_tensor_list:
                flags.append(self.viable_columns(data_array))

            # Find the intersection of viable columns across all datasets
            flags = np.prod(flags,axis=0)

            # If no columns are viable, alert user and raise exception
            if ~flags.all():
                Exception("No channels are valid across all datasets.")

            # Update the montage channel list
            self.montage_channels = self.montage_channels[flags]

            return [iarr[:,flags] for iarr in self.input_tensor_list]

    def viable_data(self,data_array):
        
        # Loop over the index associated with the datasets and keep only datasets without NaNs
        if ~np.isnan(data_array).any():
            return False
        else:
            return True

    def viable_columns(self,data_array):

        # Loop over the index associated with the columns and only return columns without NaNs
        keep_index = []
        for i_index in range(data_array.shape[1]):
            idata = data_array[:,i_index]
            if ~np.isnan(idata).any():
                keep_index.append(True)
            else:
                keep_index.append(False)

        return keep_index
    
    def consecutive_counter(self,iarr,ival):

        # Add padding to the array to get an accurate difference between elements that includes first and last
        if ~np.isnan(ival):
            diffs = np.diff(np.concatenate(([False], iarr == ival, [False])).astype(int))
        else:
            diffs = np.diff(np.concatenate(([False], np.isnan(iarr), [False])).astype(int))

        # Get the indices that denote where the array changes to/from expected value
        left_ind  = np.flatnonzero(diffs == 1)
        right_ind = np.flatnonzero(diffs == -1)

        # Get the counts of values that match the criteria
        counts = right_ind-left_ind

        # Create a mask and iterate over counts and change mask to true for consecutive values equal to or less than the allowed threshold
        mask = np.zeros(iarr.size).astype('bool')
        for ival in counts:
            if ival <= self.args.n_interp:
                for ii in range(self.args.n_interp):
                    mask[left_ind+ii] = True
        return mask

    def interpolate_data(self,data_array):

        # Loop over the columns
        for icol in range(data_array.shape[1]):
            
            # Get the current timeseries and calculate the mask
            vals = data_array[:,icol]
            mask = self.consecutive_counter(vals,np.nan)

            # Ensure that the first and last indices are false. This is to a avoid extrapolation.
            mask[0]  = False
            mask[-1] = False

            # Interpolate where appropriate
            x_vals          = np.arange(vals.size)
            x_interpretable = x_vals[~mask]
            y_interpretable = vals[~mask]
            interp_vals     = np.interp(x_vals,x_interpretable,y_interpretable)
            vals[mask]      = interp_vals[mask]
            
            # Insert new values into the original data array
            data_array[:,icol] = vals
        return data_array

class tensor_manager:

    def __init__(self):

        self.input_tensor_list = []

    def update_tensor_list(self,data):

        self.input_tensor_list.append(data)

    def create_tensor(self):

        # Clean up the data before going to the tensor
        input_array = data_viability.__init__(self)

        # Create the tensor
        self.input_tensor_dataset = [torch.utils.data.DataLoader(dataset) for dataset in input_array]

class data_manager(data_loader, channel_mapping, dataframe_manager, channel_clean, channel_montage, tensor_manager, data_viability):

    def __init__(self, infiles, args):
        """
        Initialize parent class for data loading.
        Store pathing for different data type loads.

        Args:
            infile (str): path to datafile that needs to be loaded
        """

        # Make args visible across inheritance
        self.args = args

        # Initialize the tensor list so it can be updated with each file
        tensor_manager.__init__(self)

        # Loop over files to read and store each ones data
        cnt = len(infiles)
        for ifile in infiles:
            
            # Save current file
            self.infile = ifile
            
            # Case statement the workflow
            if self.args.dtype == 'EDF':
                try:
                    self.edf_handler()
                except OSError:
                    cnt -= 1
        tensor_manager.create_tensor(self)
        print("Processed %04d files." %(cnt))

    def edf_handler(self):
        """
        Run pipeline to load EDF data.
        """

        # Import data into memory
        data_loader.load_edf(self)

        # Clean the channel names
        channel_clean.__init__(self)

        # Get the correct channels for this merger
        channel_mapping.__init__(self,self.args.channel_list)

        # Create the dataframe for the object with the cleaned labels
        dataframe_manager.__init__(self)
        dataframe_manager.column_subsection(self,self.channel_map_out)

        # Put the data into a specific montage
        channel_montage.__init__(self)

        # Update the tensor list
        tensor_manager.update_tensor_list(self,self.montaged_dataframe.values)

class CustomFormatter(argparse.HelpFormatter):
    """
    Custom formatting class to get a better argument parser help output.
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)

def make_help_str(idict):
    """
    Make a well-formated help string for the possible keyword mappings

    Args:
        idict (dict): Dictionary containing the allowed keywords values and their explanation.

    Returns:
        str: Formatted help string
    """

    return "\n".join([f"{key:15}: {value}" for key, value in idict.items()])

def input_with_tab_completion(prompt):
    def complete(text, state):
        return (file for file in readline.get_completions() if file.startswith(text))

    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")

    return input(prompt)

if __name__ == "__main__":

    # Define the allowed keywords a user can input
    allowed_input_args     = {'CSV' : 'Use a comma separated file of files to read in. (default)',
                              'MANUAL' : "Manually enter filepaths.",
                              'GLOB' : 'Use Python glob to select all files that follow a user inputted pattern.'}
    allowed_dtype_args     = {'EDF': "EDF file formats. (default)"}
    allowed_channel_args   = {'HUP1020': "Channels associated with a 10-20 montage performed at HUP.",
                              'RAW': "Use all possible channels. Warning, channels may not match across different datasets."}
    allowed_montage_args   = {'HUP1020': "Use a 10-20 montage.",
                              'COMMON_AVERAGE': "Use a common average montage."}
    allowed_viability_args = {'VIABLE_DATA': "Drop datasets that contain a NaN column. (default)",
                              'VIABLE_COLUMNS': "Use the minimum cross section of columns across all datasets that contain no NaNs."}
    
    # Make a useful help string for each keyword
    allowed_input_help     = make_help_str(allowed_input_args)
    allowed_dtype_help     = make_help_str(allowed_dtype_args)
    allowed_channel_help   = make_help_str(allowed_channel_args)
    allowed_montage_help   = make_help_str(allowed_montage_args)
    allowed_viability_help = make_help_str(allowed_viability_args)

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.", formatter_class=CustomFormatter)
    parser.add_argument("--input", choices=list(allowed_input_args.keys()), default="CSV", help=f"R|Choose an option:\n{allowed_input_help}")
    parser.add_argument("--dtype", choices=list(allowed_dtype_args.keys()), default="EDF", help=f"R|Choose an option:\n{allowed_dtype_help}")
    parser.add_argument("--channel_list", choices=list(allowed_channel_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_channel_help}")
    parser.add_argument("--montage", choices=list(allowed_montage_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_montage_help}")
    parser.add_argument("--viability", choices=list(allowed_viability_args.keys()), default="VIABLE_DATA", help=f"R|Choose an option:\n{allowed_viability_help}")
    parser.add_argument("--interp", action='store_true', default=False, help="Interpolate over NaN values of sequence length equal to n_interp.")
    parser.add_argument("--n_interp", default=1, help="Number of contiguous NaN values that can be interpolated over should the interp option be used.")
    args = parser.parse_args()

    # For testing purposes
    start = time.time()

    # Set the input file list
    if args.input == 'CSV':
        
        # Tab completion enabled input
        completer = PathCompleter()
        file_path = prompt("Please enter path to input file csv: ", completer=completer)

        # Due to the different ways paths can be inputted, using a filepointer to clean each entry best we can
        fp    = open(file_path,'r')
        files = []
        data  = fp.readline()
        while data:
            clean_data = data.replace('\n', '')
            clean_data = clean_data.split(',')
            for ival in clean_data:
                if ival != '':
                    files.append(ival)
            data = fp.readline()
    elif args.input == 'GLOB':

        # Tab completion enabled input
        completer = PathCompleter()
        #file_path = prompt("Please enter (wildcard enabled) path to input files: ", completer=completer)
        file_path = "/Users/bjprager/Documents/GitHub/SCALP_CONCAT-EEG/user_data/sample_data/edf/teug/a*/*edf"
        files     = glob.glob(file_path)

    # Load the parent class
    DM = data_manager(files, args)

    # For testing purposes
    print("Time taken in seconds: %f" %(time.time()-start))
    max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Max memory usage in GB: ~%f" %(max_mem/1e9))