# General libraries
import numpy as np
import pandas as PD

# Import the classes
from .metadata_handler import *
from .target_loader import *
from .data_loader import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .output_manager import *
from .data_viability import *

class channel_mapping:
    """
    Class devoted to the different channel mappings of interest. 

    New functions should look for the cross section of their mapping to the self.clean_channel_map data.

    Output should be a new list of channels called self.channel_map_out. Also required are the indices of intersection. (This is to update the metadata properly)
    """

    def __init__(self):
        """
        Logic gates for which channel mapping methodology to use.

        Args:
            clean_method (str, optional): Mapping method to use, see --help for complete list.
        """

    def pipeline(self,channel_mapping):
        """
        Method for working within the larger pipeline environment to get channel mappings.

        Args:
            channel_mapping (str): String that defines the logic for which mapping to use.
        """

        # Store the argument of chanel mapping to class instance
        self.channel_mapping = channel_mapping

        # Apply mapping logic
        self.mapping_logic()

        # Update the metadata
        metadata_handler.set_channels(self,self.channel_map_out)
        metadata_handler.set_sampling_frequency(self,self.metadata[self.file_cntr]['fs'][self.channel_map_out_inds])

    def direct_inputs(self,channels,channel_mapping):
        """
        Method for getting channel mappings directly outside of the pipeline environment.

        Args:
            channels (list): List of channel names to perform mapping on.
            channel_mapping (str): String that defines the logic for which mapping to use.
        """

        # Store the argument of chanel mapping to class instance
        self.clean_channel_map = channels
        self.channel_mapping   = channel_mapping

        # Apply mapping logic
        self.mapping_logic()

        return self.channel_map_out

    def mapping_logic(self):
        """
        Logic gates for the different mapping options.
        """

        # Logic for different mappings
        if self.channel_mapping == "HUP1020":
            self.mapping_HUP_1020()

    ###################################
    #### User Provided Logic Below ####
    ###################################

    def mapping_HUP_1020(self):
        """
        Mapping used to eventually build a 1020 model using HUP data.
        """

        self.master_channel_list  = ['C03', 'C04', 'CZ', 'F03', 'F04', 'F07', 'F08', 'FZ', 'FP01', 'FP02', 'O01',
                                    'O02', 'P03', 'P04', 'T03', 'T04', 'T05', 'T06']
        self.channel_map_out      = np.intersect1d(self.clean_channel_map,self.master_channel_list)
        self.channel_map_out_inds = np.where(np.isin(self.clean_channel_map, self.channel_map_out))[0]