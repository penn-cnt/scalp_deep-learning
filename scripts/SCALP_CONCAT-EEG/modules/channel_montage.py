# General libraries
import numpy as np
import pandas as PD

# Import the classes
from .data_loader import *
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .output_manager import *
from .data_viability import *

class channel_montage:
    """
    Class devoted to calculating the montages of interest for the data.
    """

    def __init__(self):
        """
        Logic gates for which montage methodology to use.
        """
        
        # Logic for different montages
        if self.args.montage == "HUP1020":
            montage_data = self.montage_HUP_1020()

        return montage_data

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

        # Update the metadata to note the montage channels
        self.metadata[self.file_cntr]['montage_channels'] = self.montage_channels

        # Pass the data to the dataframe class function for montages
        return montage_data