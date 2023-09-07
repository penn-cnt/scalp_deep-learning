# General libraries
import numpy as np
import pandas as PD

# Import the classes
from .data_loader import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .tensor_manager import *
from .data_viability import *

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