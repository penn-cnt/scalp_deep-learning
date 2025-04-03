import numpy as np
import pandas as PD
from tqdm import tqdm
import multiprocessing
from sys import argv,exit

class marsh_rejection:
    """
    Applies a marsh rejection mask to a dataframe. 
    Looks for dt=-1 from the pipeline manager to reference against the full file.
    """

    def __init__(self,DF,channels,multithread,ncpu):

        # Save the input data to class instance
        self.DF          = DF
        self.channels    = channels
        self.multithread = multithread
        self.ncpu        = ncpu
        self.bar_frmt    = '{l_bar}{bar}| {n_fmt}/{total_fmt}|'

        # Find the channel labels
        self.ref_cols     = np.setdiff1d(self.DF.columns, self.channels)
        self.merge_labels = np.concatenate((['file', 'method', 'tag'],self.channels))

    def workflow(self):

        if self.multithread:

            # Make the keys to break up
            marsh_lookup_dict = self.DF.groupby(['file']).indices
            marsh_lookup_keys = list(marsh_lookup_dict.keys())

            print(marsh_lookup_dict)
            exit()

            # Make the initial subset proposal
            subset_size  = len(marsh_lookup_keys) // self.ncpu
            list_subsets = [marsh_lookup_keys[i:i + subset_size] for i in range(0, subset_size*self.ncpu, subset_size)]

            # Handle leftovers
            remainder = list_subsets[self.ncpu*subset_size:]
            for idx,ival in enumerate(remainder):
                list_subsets[idx] = np.concatenate((list_subsets[idx],np.array([ival])))

            # Convert to indices
            list_subsets_indices = [[] for idx in range(len(list_subsets))]
            for idx,subset in enumerate(list_subsets):
                for ifile in subset:
                    list_subsets_indices[idx].extend(marsh_lookup_dict[ifile])

            # Create processes and start workers
            processes   = []
            manager     = multiprocessing.Manager()
            return_dict = manager.dict()
            for worker_id, data_chunk in enumerate(list_subsets_indices):
                process = multiprocessing.Process(target=self.calculate_marsh, args=(worker_id,data_chunk,return_dict))
                processes.append(process)
                process.start()

            # Wait for all processes to complete
            for process in processes:
                process.join()
        else:
            marsh_keys  = list(self.DF.index)
            return_dict = self.calculate_marsh(0,marsh_keys,{})
        
        # Reformat the output
        self.DF = PD.concat(return_dict.values()).reset_index(drop=True)

        return self.DF

    def calculate_marsh(self,worker_num, DF_inds, return_dict):

        try:
            # Get the data slice to work on
            current_DF = self.DF.loc[DF_inds]

            # Make a dataslice just for rms and just for ll
            DF_rms = current_DF.loc[current_DF.method=='rms']
            DF_ll  = current_DF.loc[current_DF.method=='line_length']

            # Convert the data types to numeric
            for ichannel in self.channels:
                DF_rms.loc[:,ichannel] = DF_rms[ichannel].astype('float32')
                DF_ll.loc[:,ichannel]  = DF_ll[ichannel].astype('float32')

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

            # Apply the filter for each group
            for ifile in tqdm(DF_rms_mean.index, desc='Applying Marsh Filter', total=len(DF_rms_mean.index),bar_format=self.bar_frmt, position=worker_num, leave=False, dynamic_ncols=True):
                
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
            current_DF = PD.concat((current_DF,DF_rms)).reset_index(drop=True)
            current_DF = PD.concat((current_DF,DF_ll)).reset_index(drop=True)

            # Save the results to the output object
            return_dict[worker_num] = current_DF

            if not self.multithread:
                return return_dict
        except Exception as e:

            print(DF_rms[self.channels].dtypes)

            import os,sys
            fname       = os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1]
            error_type  = sys.exc_info()[0]
            line_number = sys.exc_info()[2].tb_lineno
            print(f"Error {error_type} in line {line_number}.")
            exit()
