import numpy as np
import pandas as PD
from sys import argv,exit
from collections import Counter

class yasa_reformat:

    def __init__(self,DF,channels):

        # Save input data to instance
        self.channels = channels

        # Make a yasa lookup df slice
        self.YASA_DF = DF.loc[(DF.method == 'yasa_sleep_stage')&(DF.t_window==300)]

        # Save the data slice to update
        self.DF = DF.drop(self.YASA_DF.index)

    def workflow(self):
        self.cleanup()
        self.reformat()
        return self.DF

    def cleanup(self):

        # Create the mapping for cleanup
        new_map = {'N1':'S','N2':'S','N3':'S','R':'S','W':'W'}

        # Mapping function
        def replace_stages(x):
            for ikey in new_map.keys():
                x=x.replace(ikey,new_map[ikey])
            return x
        
        def consensus_stage(x):
            time_list = x.split('|')
            for idx,itime in enumerate(time_list):
                vals           = itime.split(',')
                count          = Counter(vals)
                time_list[idx] = max(count, key=count.get)
            return time_list
            
        # Loop over the lookup table for cleanup
        cleaned_output = []
        for i_index in self.YASA_DF.index:

            # Get the full row slice so we can build the new output with modified times
            original_slice = self.YASA_DF.loc[i_index]

            for ichannel in self.channels[:1]:

                # grab the entry to modify
                input_yasa = original_slice[ichannel]

                # Update the mapping labels in the lookup table
                input_yasa = replace_stages(input_yasa)

                # get the consensus
                output_yasa = consensus_stage(input_yasa)

                # Get the time offsets for the new rows
                dt = 30*np.arange(len(output_yasa))

                # Make the new outputs
                for idx,time_offset in enumerate(dt):
                    
                    # Update the entries with the new timing and sleep stage
                    new_slice = original_slice.copy()
                    new_slice['t_start'] += time_offset
                    new_slice['t_end']    = new_slice['t_start']+30
                    new_slice[self.channels] = output_yasa[idx]

                    # Store the results to the output array
                    cleaned_output.append(new_slice.values)
        
        # YASA dataframe creation and cleanup
        self.YASA_DF            = PD.DataFrame(cleaned_output,columns=self.YASA_DF.columns)
        self.YASA_DF['t_start'] = self.YASA_DF['t_start'].astype('int16')
        self.YASA_DF['t_end']   = self.YASA_DF['t_end'].astype('int16')
        self.YASA_DF            = self.YASA_DF.drop_duplicates(subset=['file','t_start'],keep='first').reset_index(drop=True)

    def reformat(self):

        # Get the indices for the lookup groups
        YASA_lookup_dict = self.YASA_DF.groupby(['file','t_start','t_end']).indices

        # Loop over the keys
        for ikey in YASA_lookup_dict.keys():

            # Get the value to propagate
            newval = self.YASA_DF.loc[YASA_lookup_dict[ikey]][self.channels[0]].values[0]

            # Get the indices to update
            slice_logic = (self.DF.file==ikey[0])&(self.DF.t_start>=ikey[1])&(self.DF.t_start<ikey[2])&(self.DF.method=='yasa_sleep_stage')

            # Get the dataslice from the main dataframe
            self.DF.loc[slice_logic,self.channels] = newval

                
