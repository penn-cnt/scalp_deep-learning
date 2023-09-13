# General libraries
import re
import numpy as np
import pandas as PD

# Import the classes
from .data_loader import *
from .channel_mapping import *
from .dataframe_manager import *
from .channel_montage import *
from .output_manager import *
from .data_viability import *

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