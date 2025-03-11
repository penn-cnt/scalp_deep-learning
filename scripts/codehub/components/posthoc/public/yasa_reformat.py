import numpy as np
import pandas as PD
from tqdm import tqdm
import multiprocessing
from sys import argv,exit
from collections import Counter

class yasa_reformat:

    def __init__(self,DF,channels,multithread,ncpu):

        # Save input data to instance
        self.channels    = channels
        self.multithread = multithread
        self.ncpu        = ncpu
        self.bar_frmt    = '{l_bar}{bar}| {n_fmt}/{total_fmt}|'

        # Make a yasa lookup df slice
        self.YASA_DF = DF.loc[(DF.method == 'yasa_sleep_stage')&(DF.t_window==300)]

        # Save the data slice to update
        self.DF = DF.drop(self.YASA_DF.index)

    def workflow(self):

        # Clean the lookup data
        self.cleanup()

        # Get the indices for the lookup groups
        self.YASA_lookup_dict = self.YASA_DF.groupby(['file','t_start','t_end']).indices
        YASA_keys             = list(self.YASA_lookup_dict.keys())

        if self.multithread:

            # Make the initial subset proposal
            subset_size  = len(YASA_keys) // self.ncpu
            list_subsets = [YASA_keys[i:i + subset_size] for i in range(0, subset_size*self.ncpu, subset_size)]

            # Handle leftovers
            remainder = list_subsets[self.ncpu*subset_size:]
            for idx,ival in enumerate(remainder):
                list_subsets[idx] = np.concatenate((list_subsets[idx],np.array([ival])))

            # Create processes and start workers
            processes   = []
            manager     = multiprocessing.Manager()
            return_dict = manager.dict()
            for worker_id, data_chunk in enumerate(list_subsets):
                process = multiprocessing.Process(target=self.reformat, args=(worker_id,data_chunk,return_dict))
                processes.append(process)
                process.start()

            # Wait for all processes to complete
            for process in processes:
                process.join()
        else:
            return_dict = self.reformat(0,YASA_keys,{})
        
        # Reformat the output
        self.DF = PD.concat(return_dict.values()).reset_index(drop=True)
        
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
                try:
                    output_yasa = consensus_stage(input_yasa)
                except:
                    print(input_yasa)
                    exit()

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

    def reformat(self,worker_num,inkeys,return_dict):

        # Loop over the keys
        output = []
        for ikey in tqdm(inkeys, desc='Applying YASA Restructure', total=len(inkeys),bar_format=self.bar_frmt, position=worker_num, leave=False, dynamic_ncols=True):

            # Get the value to propagate
            newval = self.YASA_DF.loc[self.YASA_lookup_dict[ikey]][self.channels[0]].values[0]

            # Get the indices to update
            base_slice     = (self.DF.file==ikey[0])&(self.DF.t_start>=ikey[1])&(self.DF.t_start<ikey[2])
            yasa_slice     = base_slice&(self.DF.method=='yasa_sleep_stage')
            non_yasa_slice = base_slice&(self.DF.method!='yasa_sleep_stage')

            # Get the dataslice from the main dataframe
            iDF = self.DF.loc[yasa_slice]
            jDF = self.DF.loc[non_yasa_slice]

            # Put the new values into the slice
            iDF.loc[:,self.channels] = newval

            # Store the results to the output
            output.append(PD.concat((iDF,jDF)))

        # Merge all the results for this worker
        return_dict[worker_num] = PD.concat(output)

        # Return the return dict if not multiprocessing
        if not self.multithread:
            return return_dict
        

                
