# Libraries to help path complete raw inputs
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# General libraries
import os
import sys
import glob
import uuid
import time
import argparse
import datetime
import numpy as np
import pandas as PD
from tqdm import tqdm
import multiprocessing

# Import the classes
from modules.datatype_handlers import *
from modules.data_loader import *
from modules.channel_mapping import *
from modules.dataframe_manager import *
from modules.channel_clean import *
from modules.channel_montage import *
from modules.output_manager import *
from modules.data_viability import *
from modules.preprocessing import *
from modules.features import *
from configs.makeconfigs import *

class data_manager(datatype_handlers, data_loader, channel_mapping, dataframe_manager, channel_clean, channel_montage, output_manager, data_viability):

    def __init__(self, input_params, args, worker_number, barrier):
        """
        Initialize parent class for data loading.
        Store pathing for different data type loads.

        Args:
            infile (str): path to datafile that needs to be loaded
        """

        # Make args visible across inheritance
        infiles            = input_params[:,0]
        start_times        = input_params[:,1].astype('float')
        end_times          = input_params[:,2].astype('float')
        self.args          = args
        self.metadata      = {}
        self.unique_id     = uuid.uuid4()
        self.bar_frmt      = '{l_bar}{bar}| {n_fmt}/{total_fmt}|'
        self.worker_number = worker_number
        self.barrier       = barrier

        # Initialize the output list so it can be updated with each file
        output_manager.__init__(self)
        
        # File management
        self.file_manager(infiles, start_times, end_times)

        # Select valid data slices
        data_viability.__init__(self)

        # Apply preprocessing as needed
        if not args.no_preprocess_flag:
            
            # Barrier the code for better output formatting
            if self.args.multithread:
                barrier.wait()

            # Add a wait for proper progress bars
            time.sleep(self.worker_number)

            # Clean up the screen
            if self.worker_number == 0:
                sys.stdout.write("\033[H")
                sys.stdout.flush()

            # Process
            preprocessing.__init__(self)

        if not args.no_feature_flag:
            if self.args.multithread:
                barrier.wait()

            # Add a wait for proper progress bars
            time.sleep(self.worker_number)

            # Clean up the screen
            if self.worker_number == 0:
                sys.stdout.write("\033[H")
                sys.stdout.flush()
            features.__init__(self)

        # Save the results
        output_manager.save_features(self)

    def file_manager(self,infiles, start_times, end_times):
        """
        Loop over the input files and send them to the correct data handler.

        Args:
            infiles (str list): Path to each dataset
            start_times (float list): Start times in seconds to start sampling
            end_times (float list): End times in seconds to end sampling
        """

        # Intialize a variable that stores the previous filepath. This allows us to cache data and only read in as needed. (i.e. new path != old path)
        self.oldfile = None  

        # Loop over files to read and store each ones data
        nfile = len(infiles)
        desc  = "Initial load with id %s:" %(self.unique_id)
        for ii,ifile in tqdm(enumerate(infiles), desc=desc, total=nfile, bar_format=self.bar_frmt, position=self.worker_number, leave=False, disable=self.args.silent):            
        
            # Save current file info
            self.infile    = ifile
            self.t_start   = start_times[ii]
            self.t_end     = end_times[ii]
            
            # Update the metadata
            self.file_cntr = ii
            self.metadata[self.file_cntr]            = {}
            self.metadata[self.file_cntr]['file']    = self.infile
            self.metadata[self.file_cntr]['t_start'] = self.t_start
            self.metadata[self.file_cntr]['t_end']   = self.t_end
            self.metadata[self.file_cntr]['dt']      = self.t_end-self.t_start

            # Case statement the workflow
            if self.args.dtype == 'EDF':
                datatype_handlers.edf_handler(self)
                self.oldfile = self.infile


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

def parse_list(input_str):
    """
    Helper function to allow list inputs to argparse using a space or comma

    Args:
        input_str (str): Users inputted string

    Returns:
        list: Input argument list as python list
    """

    # Split the input using either spaces or commas as separators
    values = input_str.replace(',', ' ').split()
    return [int(value) for value in values]

def start_analysis(data_chunk,args,worker_id,barrier):
    """
    Helper function to allow for easy multiprocessing initialization.
    """

    DM = data_manager(data_chunk,args,worker_id,barrier)

if __name__ == "__main__":

    # Define the allowed keywords a user can input
    allowed_input_args      = {'CSV' : 'Use a comma separated file of files to read in. (default)',
                               'MANUAL' : "Manually enter filepaths.",
                               'GLOB' : 'Use Python glob to select all files that follow a user inputted pattern.'}
    allowed_dtype_args      = {'EDF': "EDF file formats. (default)"}
    allowed_channel_args    = {'HUP1020': "Channels associated with a 10-20 montage performed at HUP.",
                               'RAW': "Use all possible channels. Warning, channels may not match across different datasets."}
    allowed_montage_args    = {'HUP1020': "Use a 10-20 montage.",
                               'COMMON_AVERAGE': "Use a common average montage."}
    allowed_viability_args  = {'VIABLE_DATA': "Drop datasets that contain a NaN column. (default)",
                               'VIABLE_COLUMNS': "Use the minimum cross section of columns across all datasets that contain no NaNs."}
    
    # Make a useful help string for each keyword
    allowed_input_help     = make_help_str(allowed_input_args)
    allowed_dtype_help     = make_help_str(allowed_dtype_args)
    allowed_channel_help   = make_help_str(allowed_channel_args)
    allowed_montage_help   = make_help_str(allowed_montage_args)
    allowed_viability_help = make_help_str(allowed_viability_args)

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.", formatter_class=CustomFormatter)

    datamerge_group = parser.add_argument_group('Data Merging Options')
    datamerge_group.add_argument("--input", choices=list(allowed_input_args.keys()), default="GLOB", help=f"R|Choose an option:\n{allowed_input_help}")
    datamerge_group.add_argument("--n_input", type=int, help=f"Limit number of files read in. Useful for testing or working in batches.")
    datamerge_group.add_argument("--n_offset", type=int, default=0, help=f"Offset the files read in. Useful for testing or working in batch.")
    datamerge_group.add_argument("--dtype", choices=list(allowed_dtype_args.keys()), default="EDF", help=f"R|Choose an option:\n{allowed_dtype_help}")
    datamerge_group.add_argument("--t_start", default=0, help="Time in seconds to start data collection.")
    datamerge_group.add_argument("--t_end", default=-1, help="Time in seconds to end data collection. (-1 represents the end of the file.)")
    datamerge_group.add_argument("--t_window", type=parse_list, help="List of window sizes, effectively setting multiple t_start and t_end for a single file.")
    datamerge_group.add_argument("--multithread", action='store_true', default=False, help="Multithread flag.")
    datamerge_group.add_argument("--ncpu", type=int, default=2, help="Number of CPUs to use if multithread.")

    channel_group = parser.add_argument_group('Channel label Options')
    channel_group.add_argument("--channel_list", choices=list(allowed_channel_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_channel_help}")

    montage_group = parser.add_argument_group('Montage Options')
    montage_group.add_argument("--montage", choices=list(allowed_montage_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_montage_help}")

    viability_group = parser.add_argument_group('Data viability Options')
    viability_group.add_argument("--viability", choices=list(allowed_viability_args.keys()), default="VIABLE_DATA", help=f"R|Choose an option:\n{allowed_viability_help}")
    viability_group.add_argument("--interp", action='store_true', default=False, help="Interpolate over NaN values of sequence length equal to n_interp.")
    viability_group.add_argument("--n_interp", default=1, help="Number of contiguous NaN values that can be interpolated over should the interp option be used.")

    preprocessing_group = parser.add_argument_group('Preprocessing Options')
    preprocessing_group.add_argument("--no_preprocess_flag", action='store_true', default=False, help="Do not run preprocessing on data.")
    preprocessing_group.add_argument("--preprocess_file", help="Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.")

    feature_group = parser.add_argument_group('Feature Extraction Options')
    feature_group.add_argument("--no_feature_flag", action='store_true', default=False, help="Do not run feature extraction on data.")
    feature_group.add_argument("--feature_file", help="Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.")

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--outdir", default="../../user_data/derivative/", help="Output directory.") 

    misc_group = parser.add_argument_group('Misc Options')
    misc_group.add_argument("--silent", action='store_true', default=False, help="Silent mode.")
    args = parser.parse_args()

    # Set the input file list
    if args.input == 'CSV':
        
        # Tab completion enabled input
        completer = PathCompleter()
        print("Using CSV input. Enter a three column csv file with filepath,starttime,endtime.")
        print("If not starttime or endtime provided, defaults to argument inputs. Use --help for more information.")
        file_path = prompt("Please enter path to input file csv: ", completer=completer)

        # Read in csv file
        input_csv   = PD.read_csv("./sample_input.csv")
        files       = input_csv['filepath'].values
        start_times = input_csv['start_time'].values
        end_times   = input_csv['end_time'].values

        # Replace NaNs with appropriate times as needed
        start_times = np.nan_to_num(start_times,nan=args.t_start)
        end_times   = np.nan_to_num(end_times,nan=args.t_end)
    elif args.input == 'GLOB':

        # Tab completion enabled input
        completer = PathCompleter()
        file_path = prompt("Please enter (wildcard enabled) path to input files: ", completer=completer)
        files     = glob.glob(file_path)

        # Create start and end times array
        start_times = args.t_start*np.ones(len(files))
        end_times   = args.t_end*np.ones(len(files))
    else:
        # Tab completion enabled input
        completer = PathCompleter()
        file_path = prompt("Please enter path to input file: ", completer=completer)
        files     = [file_path]

        # Create start and end times array
        start_times = args.t_start*np.ones(len(files))
        end_times   = args.t_end*np.ones(len(files))

    # Apply any file offset as needed
    files       = files[args.n_offset:]
    start_times = start_times[args.n_offset:]
    end_times   = end_times[args.n_offset:]

    # Limit file length as needed
    if args.n_input != None:
        files       = files[:args.n_input]
        start_times = start_times[:args.n_input]
        end_times   = end_times[:args.n_input]

    # If using a sliding time window, duplicate inputs with the correct inputs
    if args.t_window != None:
        new_files = []
        new_start = []
        new_end   = []
        for ifile in files:

            # Read in just the header to get duration
            if args.t_end == -1:
                t_end = highlevel.read_edf_header(ifile)['Duration']
            else:
                t_end = args.t_end

            # Get the start time for the windows
            if args.t_start == None:
                t_start = 0
            else:
                t_start = args.t_start

            for iwindow in args.t_window:
                
                # Get the list of windows start and end times
                windowed_start = np.array(range(t_start,t_end,iwindow))
                windowed_end   = np.array(range(t_start+iwindow,t_end+iwindow,iwindow))

                # Make sure we have no values outside the right range
                windowed_end[(windowed_end>t_end)] = t_end

                # Loop over the new entries and tile the input lists as needed
                for idx,istart in enumerate(windowed_start):
                    new_files.append(ifile)
                    new_start.append(istart)
                    new_end.append(windowed_end[idx])
        files       = new_files
        start_times = new_start
        end_times   = new_end 

    # Make configuration files as needed
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args.preprocess_file == None and not args.no_preprocess_flag:
        from modules import preprocessing
        args.preprocess_file = "configs/preprocessing_"+timestamp+".yaml"
        config_handler       = make_config(preprocessing,args.preprocess_file)
    if args.feature_file == None:
        from modules import features
        args.feature_file = "features_"+timestamp+".yaml"
        config_handler    = make_config(features,args.feature_file)

    # Multithread options
    input_parameters = np.column_stack((files, start_times, end_times))
    if args.multithread:

        # Calculate the size of each subset based on the number of processes
        subset_size  = input_parameters.shape[0] // args.ncpu
        list_subsets = [input_parameters[i:i + subset_size] for i in range(0, input_parameters.shape[0], subset_size)]

        # Handle leftovers
        if len(list_subsets) > args.ncpu:
            arr_ncpu  = list_subsets[args.ncpu-1]
            arr_ncpu1 = list_subsets[args.ncpu]

            list_subsets[args.ncpu-1] = np.concatenate((arr_ncpu,arr_ncpu1), axis=0)
            list_subsets.pop(-1)

        # Create a barrier for synchronization
        barrier = multiprocessing.Barrier(args.ncpu)

        # Create processes and start workers
        processes = []
        for worker_id, data_chunk in enumerate(list_subsets):
            process = multiprocessing.Process(target=start_analysis, args=(data_chunk,args,worker_id,barrier))
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        for process in processes:
            process.join()
    else:
        start_analysis(input_parameters, args, 0, None)