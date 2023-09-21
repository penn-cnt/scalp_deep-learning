# General libraries
import numpy as np
import pandas as PD
from  pyedflib import highlevel

# Import the classes
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .output_manager import *
from .data_viability import *

class data_loader:
    """
    Class devoted to loading in raw data into the shared class instance.
    """

    def load_edf(self):
        """
        Parent class data loader for EDF file format.
        """

        # Load current edf data into memory
        self.raw_data, channel_metadata, scan_metadata = highlevel.read_edf(self.infile)
        self.channels = highlevel.read_edf_header(self.infile)['channels']

        # Clean up the edf data
        self.channels = [ichannel.upper() for ichannel in self.channels]
        
        # Make a metadata dataframe in case we need to store information through transformations
        self.metadata           = PD.DataFrame(columns=self.channels)
        sample_frequency        = [ichannel['sample_frequency'] for ichannel in channel_metadata]
        self.metadata.loc['fs'] = sample_frequency

        # Get only the time slices of interest
        for ii,isamp in enumerate(sample_frequency):
            
            # Calculate the index of the start
            samp_start = isamp*self.t_start

            # Calculate the index of the end
            if self.t_end == -1:
                samp_end = len(self.raw_data[ii])
            else:
                samp_end = isamp*self.t_end
            


        print(self.raw_data[0])
        print(self.raw_data[0].shape)
        import sys
        sys.exit()

        # Get the underlying data shapes
        self.ncol = len(self.raw_data)
        self.nrow = max([ival.size for ival in self.raw_data])