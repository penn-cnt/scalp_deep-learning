# General libraries
import numpy as np
import pandas as PD

# Import the classes
from .data_loader import *
from .channel_mapping import *
from .channel_clean import *
from .channel_montage import *
from .tensor_manager import *
from .data_viability import *

class dataframe_manager:
    """
    Class devoted to all things dataframe related. These dataframes will be used to populate the PyTorch tensors.
    """

    def __init__(self):

        self.dataframe = PD.DataFrame(index=range(self.nrow), columns=self.master_channel_list)
        values         = self.dataframe.values
        for idx,icol in enumerate(self.clean_channel_map):
            ivals      = self.raw_data[idx]
            try:
                column_idx                     = np.argwhere(self.dataframe.columns==icol)[0][0]
                values[:ivals.size,column_idx] = ivals
            except IndexError:
                pass

    def column_subsection(self,keep_columns):
        """
        Return a dataframe with only the columns requested.

        Args:
            keep_columns (list of channel labels): List of columns to keep.
        """

        # Get the columns to drop
        drop_cols = np.setdiff1d(self.dataframe.columns,keep_columns)
        self.dataframe = self.dataframe.drop(drop_cols, axis=1)

    def montaged_dataframe(self,data,columns):
        """
        Create a dataframe that stores the montaged data.

        Args:
            data (array): array of montaged data
            columns (list): List of column names
        """

        self.montaged_dataframe = PD.DataFrame(data,columns=columns)