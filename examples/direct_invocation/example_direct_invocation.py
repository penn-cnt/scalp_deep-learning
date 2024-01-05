# Local imports
from modules.addons.data_loader import *
from modules.addons.channel_clean import *
from modules.addons.channel_mapping import *
from modules.addons.channel_montage import *

class data_handler:

    def __init__(self):
        pass

    def data_prep(self):

        # Create pointers to the relevant classes
        DL    = data_loader()
        CHCLN = channel_clean()
        CHMAP = channel_mapping()
        CHMON = channel_montage()

        # Get the raw data and pointers
        DF,self.fs = DL.direct_inputs(self.infile,'edf')

        # Get the cleaned channel names
        clean_channels = CHCLN.direct_inputs(DF.columns)
        channel_dict   = dict(zip(DF.columns,clean_channels))
        DF.rename(columns=channel_dict,inplace=True)

        # Get the channel mapping
        channel_map = CHMAP.direct_inputs(DF.columns,"HUP1020")
        DF          = DF[channel_map]

        # Get the montage
        self.DF = CHMON.direct_inputs(DF,"HUP1020")

        # Read in optional sleep wake power data if provided
        if self.args.sleep_wake_power != None:
            self.read_sleep_wake_data()
        else:
            self.t_flag = False