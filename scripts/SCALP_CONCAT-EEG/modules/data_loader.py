# General libraries
import numpy as np
import pandas as PD
from  pyedflib import highlevel

# CNT/EEG Specific
from ieeg.auth import Session

# Import the classes
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .output_manager import *
from .data_viability import *

class data_loader:
    """
    Class devoted to loading in raw data into the shared class instance.

    New functions should make use of the specific raw data handler for their dataset.
    """

    def raw_dataslice(self,sample_frequency,majoraxis='columnar'):

        # Get only the time slices of interest
        self.raw_data = []
        for ii,isamp in enumerate(sample_frequency):
            
            # Calculate the index of the start
            samp_start = int(isamp*self.t_start)

            # Calculate the index of the end
            if self.t_end == -1:
                samp_end = int(len(self.indata[ii]))
            else:
                samp_end = int(isamp*self.t_end)

            if majoraxis == 'columnar':
                self.raw_data.append(self.indata[samp_start:samp_end,ii])
            elif majoraxis == 'row':
                self.raw_data.append(self.indata[ii][samp_start:samp_end])

        # Get the underlying data shapes
        self.ncol = len(self.raw_data)
        self.nrow = max([ival.size for ival in self.raw_data])        

    def load_edf(self):
        """
        Load EDF data directly into the pipeline.
        """

        # Load current edf data into memory
        if self.infile != self.oldfile:
            self.indata, self.channel_metadata, scan_metadata = highlevel.read_edf(self.infile)
            self.channels = [ival['label'] for ival in self.channel_metadata]

        # Save the channel names
        self.channels                             = [ichannel.upper() for ichannel in self.channels]
        self.metadata[self.file_cntr]['channels'] = self.channels

        # Calculate the sample frequencies to save the information and make time cuts
        sample_frequency                    = np.array([ichannel['sample_frequency'] for ichannel in self.channel_metadata])
        self.metadata[self.file_cntr]['fs'] = sample_frequency

        # Get the rawdata
        self.raw_data(sample_frequency)

    def load_iEEG(self,username,password,dataset_name):

        # Load current data into memory
        if self.infile != self.oldfile:
            with Session(username,password) as session:
                dataset     = session.open_dataset(dataset_name)
                channels    = dataset.ch_labels
                self.indata = dataset.get_data(0,np.inf,range(len(channels)))
            session.close_dataset(dataset_name)
        
        # Save the channel names to metadata
        self.channels = channels
        self.metadata[self.file_cntr]['channels'] = self.channels
        
        # Calculate the sample frequencies
        sample_frequency                    = [dataset.get_time_series_details(ichannel).sample_rate for ichannel in self.channels]
        self.metadata[self.file_cntr]['fs'] = sample_frequency

            

