from  pyedflib import highlevel

class data_loader

    def __init__(self, infile):
        self.infile = infile
    
    def load_edf(self):
        """
        Load data from an edf 
        """

        self.raw_data, self.raw_channel, self.metadata = highlevel.read_edf(self.infile)
        channels = highlevel.read_edf_header(self.infile)['channels']