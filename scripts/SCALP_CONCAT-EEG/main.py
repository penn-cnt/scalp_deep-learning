# Libraries to help path complete raw inputs
from pathlib import Path
from prompt_toolkit import prompt
from prompt_toolkit.completion import PathCompleter

# General libraries
import sys
import glob
import time
import resource
import argparse
import datetime
import numpy as np
import pandas as PD

# Import the classes
from modules.data_loader import *
from modules.channel_mapping import *
from modules.dataframe_manager import *
from modules.channel_clean import *
from modules.channel_montage import *
from modules.output_manager import *
from modules.data_viability import *
from modules.preprocessing import *
from configs.makeconfigs import *

class data_manager(data_loader, channel_mapping, dataframe_manager, channel_clean, channel_montage, output_manager, data_viability):

    def __init__(self, infiles, start_times, end_times, args):
        """
        Initialize parent class for data loading.
        Store pathing for different data type loads.

        Args:
            infile (str): path to datafile that needs to be loaded
        """

        # Make args visible across inheritance
        self.args = args

        # Initialize the output list so it can be updated with each file
        output_manager.__init__(self)
        
        # File management
        self.file_manager(infiles, start_times, end_times)

        # Select valid data slices
        data_viability.__init__(self)

        # Apply preprocessing as needed
        if not args.no_preprocess_flag:
            preprocessing.__init__(self)
        
        # Save the intermediate results
        output_manager.save_output_list(self)

    def file_manager(self,infiles, start_times, end_times):
        """
        Loop over the input files and send them to the correct data handler.

        Args:
            infiles (str list): Path to each dataset
            start_times (float list): Start times in seconds to start sampling
            end_times (float list): End times in seconds to end sampling
        """

        # Loop over files to read and store each ones data
        file_cnt = len(infiles)
        for ii,ifile in enumerate(infiles):
            
            # Save current file info
            self.infile  = ifile
            self.t_start = start_times[ii]
            self.t_end   = end_times[ii]
            
            # Case statement the workflow
            print("Reading in %s." %(self.infile))
            if self.args.dtype == 'EDF':
                try:
                    self.edf_handler()
                except OSError:
                    file_cnt -= 1        

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

        # Perform next steps only if we have a viable dataset
        if self.dataframe.shape[0] == 0:
            print("Skipping %s.\n(This could be due to poorly selected start and end times.)" %(self.infile))
            pass
        else:
            # Put the data into a specific montage
            montage_data = channel_montage.__init__(self)
            dataframe_manager.montaged_dataframe(self,montage_data,self.montage_channels)

            # Update the output list
            output_manager.update_output_list(self,self.montaged_dataframe.values,self.metadata)

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
    datamerge_group.add_argument("--input", choices=list(allowed_input_args.keys()), default="CSV", help=f"R|Choose an option:\n{allowed_input_help}")
    datamerge_group.add_argument("--dtype", choices=list(allowed_dtype_args.keys()), default="EDF", help=f"R|Choose an option:\n{allowed_dtype_help}")
    datamerge_group.add_argument("--channel_list", choices=list(allowed_channel_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_channel_help}")
    datamerge_group.add_argument("--montage", choices=list(allowed_montage_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_montage_help}")
    datamerge_group.add_argument("--viability", choices=list(allowed_viability_args.keys()), default="VIABLE_DATA", help=f"R|Choose an option:\n{allowed_viability_help}")
    datamerge_group.add_argument("--interp", action='store_true', default=False, help="Interpolate over NaN values of sequence length equal to n_interp.")
    datamerge_group.add_argument("--n_interp", default=1, help="Number of contiguous NaN values that can be interpolated over should the interp option be used.")
    datamerge_group.add_argument("--t_start", default=120, help="Time in seconds to start data collection.")
    datamerge_group.add_argument("--t_end", default=600, help="Time in seconds to end data collection. (-1 represents the end of the file.)")
    
    preprocessing_group = parser.add_argument_group('Preprocessing Options')
    preprocessing_group.add_argument("--no_preprocess_flag", action='store_true', default=False, help="Do not run preprocessing on data.")
    preprocessing_group.add_argument("--preprocess_file", help="Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.")

    feature_group = parser.add_argument_group('Feature Extraction Options')
    feature_group.add_argument("--no_feature_flag", action='store_true', default=False, help="Do not run feature extraction on data.")
    feature_group.add_argument("--feature_file", help="Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.")

    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--outdir", default="../../user_data/derivative/", help="Output directory.") 
    args = parser.parse_args()

    # For testing purposes
    start = time.time()

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
        #file_path = prompt("Please enter (wildcard enabled) path to input files: ", completer=completer)
        file_path = "/Users/bjprager/Documents/GitHub/SCALP_CONCAT-EEG/user_data/sample_data/edf/ieeg/sub*/*/eeg/*edf"
        files     = glob.glob(file_path)[:5]

        # Create start and end times array
        start_times = args.t_start*np.ones(len(files))
        end_times   = args.t_end*np.ones(len(files))

    # Make configuration files as needed
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    if args.preprocess_file == None and not args.no_preprocess_flag:
        from modules import preprocessing
        args.preprocess_file = "configs/preprocessing_"+timestamp+".yaml"
        config_handler       = make_config(preprocessing,args.preprocess_file)
    if args.feature_file == None:
        args.feature_file = "features_"+timestamp+".yaml"
        #config_handler    = make_config('preprocess',args.preprocess_file)

    # Load the parent class
    DM = data_manager(files, start_times, end_times, args)