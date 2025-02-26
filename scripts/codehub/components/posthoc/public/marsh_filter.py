import numpy as np
import pandas as PD
from tqdm import tqdm
from sys import argv,exit

class marsh_rejection:
    """
    Applies a marsh rejection mask to a dataframe. 
    Looks for dt=-1 from the pipeline manager to reference against the full file.
    """

    def __init__(self,DF,channels):

        # Save the input data to class instance
        self.DF       = DF
        self.channels = channels

        # Find the channel labels
        self.ref_cols     = np.setdiff1d(self.DF.columns, self.channels)
        self.merge_labels = np.concatenate((['file', 'method', 'tag'],self.channels))

    def workflow(self):
        self.calculate_marsh()
        return self.DF

    def calculate_marsh(self):

        # Make a dataslice just for rms and just for ll
        DF_rms = self.DF.loc[self.DF.method=='rms']
        DF_ll  = self.DF.loc[self.DF.method=='line_length']

        # Get the group level values
        rms_obj      = DF_rms.groupby(['file'])[self.channels]
        ll_obj       = DF_ll.groupby(['file'])[self.channels]
        DF_rms_mean  = rms_obj.mean()
        DF_rms_stdev = rms_obj.std()
        DF_ll_mean   = ll_obj.mean()
        DF_ll_stdev  = ll_obj.std()

        # Make output lists
        rms_output = []
        ll_output  = []

        # Apply the filter
        DF_rms.set_index(['file'],inplace=True)
        DF_ll.set_index(['file'],inplace=True)
        DF_rms = DF_rms.sort_values(by=['t_start','t_end','t_window'])
        DF_ll  = DF_ll.sort_values(by=['t_start','t_end','t_window'])
        for ifile in tqdm(DF_rms_mean.index,desc='Applying filter',total=len(DF_rms_mean.index)):
            
            # Get the reference values
            ref_rms_mean  = DF_rms_mean.loc[ifile]
            ref_rms_stdev = DF_rms_stdev.loc[ifile]
            ref_ll_mean   = DF_ll_mean.loc[ifile]
            ref_ll_stdev  = DF_ll_stdev.loc[ifile]
            
            # Get the rms mask
            DF_rms_slice                      = DF_rms.loc[[ifile]]
            channel_rms_marsh                 = DF_rms_slice[self.channels]/(ref_rms_mean+2*ref_rms_stdev).values
            DF_rms_slice.loc[:,self.channels] = channel_rms_marsh[self.channels].values
            DF_rms_slice.loc[:,['method']]    = 'marsh_filter'
            DF_rms_slice.loc[:,['tag']]       = 'rms'
            rms_output.append(DF_rms_slice)

            # Get the line length mask
            DF_ll_slice                      = DF_ll.loc[[ifile]]
            channel_ll_marsh                 = DF_ll_slice[self.channels]/(ref_ll_mean+2*ref_ll_stdev).values
            DF_ll_slice.loc[:,self.channels] = channel_ll_marsh[self.channels].values
            DF_ll_slice.loc[:,['method']]    = 'marsh_filter'
            DF_ll_slice.loc[:,['tag']]       = 'line_length'
            ll_output.append(DF_ll_slice)
        
        # make the output dataframes 
        DF_rms = PD.concat(rms_output)
        DF_ll  = PD.concat(ll_output)
        
        # Clean up the outputs
        DF_rms['file'] = DF_rms.index
        DF_ll['file']  = DF_ll.index
        DF_rms         = DF_rms.reset_index(drop=True)
        DF_ll          = DF_ll.reset_index(drop=True)

        # Append the results to input
        self.DF = PD.concat((self.DF,DF_rms)).reset_index(drop=True)
        self.DF = PD.concat((self.DF,DF_ll)).reset_index(drop=True)