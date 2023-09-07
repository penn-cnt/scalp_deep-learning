# General libraries
import numpy as np
import pandas as PD

# Import the classes
from data_loader import *
from channel_mapping import *
from dataframe_manager import *
from channel_clean import *
from channel_montage import *
from tensor_manager import *
from data_viability import *

class data_viability:

    def __init__(self):

        # Interpolate over small subsets of NaN if requested
        if self.args.interp:
            for idx,data_array in self.input_tensor_list:
                self.input_tensor_list[idx] = self.interpolate_data(data_array)

        # Find minimum viable datasets
        if self.args.viability == "VIABLE_DATA":

            # Loop over each dataset and find the ones that have no NaNs
            flags = []
            for idx,data_array in enumerate(self.input_tensor_list):
                flags.append(self.viable_data(data_array))
            
            # Output list
            self.viable_data = []
            for idx,iarr in enumerate(self.input_tensor_list):
                if flags[idx]:
                    self.viable_data.append(iarr)
        
        elif self.args.viability == 'VIABLE_COLUMNS':
            
            # Loop over each array and get the valid columns as boolean flag
            flags = []
            for idx,data_array in self.input_tensor_list:
                flags.append(self.viable_columns(data_array))

            # Find the intersection of viable columns across all datasets
            flags = np.prod(flags,axis=0)

            # If no columns are viable, alert user and raise exception
            if ~flags.all():
                Exception("No channels are valid across all datasets.")

            # Update the montage channel list
            self.montage_channels = self.montage_channels[flags]

            self.viable_data = [iarr[:,flags] for iarr in self.input_tensor_list]

    def viable_data(self,data_array):
        
        # Loop over the index associated with the datasets and keep only datasets without NaNs
        if ~np.isnan(data_array).any():
            return False
        else:
            return True

    def viable_columns(self,data_array):

        # Loop over the index associated with the columns and only return columns without NaNs
        keep_index = []
        for i_index in range(data_array.shape[1]):
            idata = data_array[:,i_index]
            if ~np.isnan(idata).any():
                keep_index.append(True)
            else:
                keep_index.append(False)

        return keep_index
    
    def consecutive_counter(self,iarr,ival):

        # Add padding to the array to get an accurate difference between elements that includes first and last
        if ~np.isnan(ival):
            diffs = np.diff(np.concatenate(([False], iarr == ival, [False])).astype(int))
        else:
            diffs = np.diff(np.concatenate(([False], np.isnan(iarr), [False])).astype(int))

        # Get the indices that denote where the array changes to/from expected value
        left_ind  = np.flatnonzero(diffs == 1)
        right_ind = np.flatnonzero(diffs == -1)

        # Get the counts of values that match the criteria
        counts = right_ind-left_ind

        # Create a mask and iterate over counts and change mask to true for consecutive values equal to or less than the allowed threshold
        mask = np.zeros(iarr.size).astype('bool')
        for ival in counts:
            if ival <= self.args.n_interp:
                for ii in range(self.args.n_interp):
                    mask[left_ind+ii] = True
        return mask

    def interpolate_data(self,data_array):

        # Loop over the columns
        for icol in range(data_array.shape[1]):
            
            # Get the current timeseries and calculate the mask
            vals = data_array[:,icol]
            mask = self.consecutive_counter(vals,np.nan)

            # Ensure that the first and last indices are false. This is to a avoid extrapolation.
            mask[0]  = False
            mask[-1] = False

            # Interpolate where appropriate
            x_vals          = np.arange(vals.size)
            x_interpretable = x_vals[~mask]
            y_interpretable = vals[~mask]
            interp_vals     = np.interp(x_vals,x_interpretable,y_interpretable)
            vals[mask]      = interp_vals[mask]
            
            # Insert new values into the original data array
            data_array[:,icol] = vals
        return data_array
