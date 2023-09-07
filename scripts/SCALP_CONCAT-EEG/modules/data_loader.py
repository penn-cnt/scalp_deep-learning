# General libraries
import numpy as np
import pandas as PD
from  pyedflib import highlevel

# Import the classes
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .tensor_manager import *
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
        self.raw_data, self.channel_metadata, self.scan_metadata = highlevel.read_edf(self.infile)
        self.channels = highlevel.read_edf_header(self.infile)['channels']

        # Clean up the edf data
        self.channels = [ichannel.upper() for ichannel in self.channels]

        # Get the underlying data shapes
        self.ncol = len(self.raw_data)
        self.nrow = max([ival.size for ival in self.raw_data])