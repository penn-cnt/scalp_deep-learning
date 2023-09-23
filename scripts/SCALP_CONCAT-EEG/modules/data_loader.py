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
        raw_data, channel_metadata, scan_metadata = highlevel.read_edf(self.infile)
        self.channels = highlevel.read_edf_header(self.infile)['channels']

        # Clean up the edf data
        self.channels = [ichannel.upper() for ichannel in self.channels]
        
        # Make a metadata dataframe in case we need to store information through transformations
        self.metadata              = {}
        self.metadata[self.infile] = {}
        self.metadata[self.infile]['channels'] = self.channels

        # Calculate the sample frequencies to save the information and make time cuts
        sample_frequency                 = np.array([ichannel['sample_frequency'] for ichannel in channel_metadata])
        self.metadata[self.infile]['fs'] = sample_frequency

        # Get only the time slices of interest
        self.raw_data = []
        for ii,isamp in enumerate(sample_frequency):
            
            # Calculate the index of the start
            samp_start = int(isamp*self.t_start)

            # Calculate the index of the end
            if self.t_end == -1:
                samp_end = int(len(raw_data[ii]))
            else:
                samp_end = int(isamp*self.t_end)

            # Update the raw data array to only get the relevant time slice
            self.raw_data.append(raw_data[ii][samp_start:samp_end])

        # Get the underlying data shapes
        self.ncol = len(self.raw_data)
        self.nrow = max([ival.size for ival in self.raw_data])