# General libraries
import numpy as np
import pandas as PD

# Import the classes
from .metadata_handler import *
from .target_loader import *
from .data_loader import *
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .output_manager import *
from .data_viability import *

class channel_montage:
    """
    Class devoted to calculating the montages of interest for the data.

    New functions should look for the relevant data in the self.dataframe object.

    Output should create a new object self.montage_channels with the labels for the output montage data vectors. You must also return the actual montage data.
    (This is due to an inheritance issue passing this data directly to the dataframe manager.)
    """

    def __init__(self):
        pass

    def pipeline(self):
        """
        Method for working within the larger pipeline environment to get channel montages.

        Returns:
            array: Array of montage data. (Issue with inheritance requires a direct passback and not through instance.)
        """

        # Apply the montage logic
        montage_data = self.montage_logic(self.args.montage)

        # Update the metadata to note the montage channels
        metadata_handler.set_montage_channels(self,self.montage_channels)

        return montage_data

    def direct_inputs(self,DF,montage):
        """
        Method for getting channel montages directly outside of the pipeline environment.

        Args:
            DF (datafram): Dataframe to get montage for.
            montage (str): Montage to perform

        Returns:
            dataframe: New dataframe with montage data and channel names
        """
        
        # Save the user provided dataframe
        self.dataframe = DF

        # Apply montage logic
        montage_data = self.montage_logic(montage)

        return PD.DataFrame(montage_data,columns=self.montage_channels)

    def montage_logic(self, montage):
        """
        Logic gate for how to montage the data.

        Args:
            montage (str): User provided string for type of montage to perform.

        Returns:
            array: array of montage data
        """

        # Logic for different montages
        if montage == "HUP1020":
            return self.montage_HUP_1020()
        elif montage == "COMMON_AVERAGE":
            return self.montage_common_average()        


    ###################################
    #### User Provided Logic Below ####
    ###################################

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
        self.montage_channels = [f"{ichannel}-com_avg" for ichannel in self.dataframe.columns]

        # Make the montage dataframe
        return montage_data

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
        montage_data = np.zeros((self.dataframe.shape[0],len(bipolar_array))).astype('float64')
        for ii,ival in enumerate(bipolar_array):
            try:
                montage_data[:,ii] = self.dataframe[ival[0]].values-self.dataframe[ival[1]].values
            except KeyError:
                montage_data[:,ii] = np.nan

        # Get the new montage channel labels
        self.montage_channels = [f"{ichannel[0]}-{ichannel[1]}" for ichannel in bipolar_array]

        # Pass the data to the dataframe class function for montages
        return montage_data