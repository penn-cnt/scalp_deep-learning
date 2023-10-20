import numpy as np

class metadata_handler:

    def __init__(self):

        # Initialize the metadata dictionary
        self.metadata = {}

        # Because we are using built-in lists to store each data transform, we want to index the dictionary to find the right metadata
        self.metadata[self.file_cntr] = {}

        # Create a metadata entry to store what functions were used throughout the process
        self.metadata[self.file_cntr]['history'] = []

    def update_metadata(self,inputs):
        """
        Update the metadata dictionary with new dictionary. This is useful if rejecting datasets.
        """

        self.metadata = inputs.copy()

    def update_history(self,input_method):

        self.metadata[self.file_cntr]['history'].append(input_method)

    def highlevel_info(self):
        """
        Store high level information about the current dataset and the timeslices extracted from it
        """

        self.metadata[self.file_cntr]['file']    = self.infile
        self.metadata[self.file_cntr]['t_start'] = self.t_start
        self.metadata[self.file_cntr]['t_end']   = self.t_end
        self.metadata[self.file_cntr]['dt']      = self.t_end-self.t_start

    def set_channels(self,inputs):

        self.metadata[self.file_cntr]['channels'] = inputs

    def set_montage_channels(self,inputs):

        self.metadata[self.file_cntr]['montage_channels'] = inputs

    def set_sampling_frequency(self,inputs):

        self.metadata[self.file_cntr]['fs'] = inputs

    def set_targets(self,inputs):

        self.metadata[self.file_cntr]['targets'] = inputs
