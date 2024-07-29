import argparse
from sys import exit
import pandas as PD
from pyedflib.highlevel import read_edf,read_edf_header
from components.workflows.public.channel_clean import channel_clean

class machine_level:

    def __init__(self,args):

        self.args = args

    def run_tests(self):
        try:
            self.test_header()
            #self.test_channels()
        except:
            exit(1)
        self.test_channels()

    def failure(self):
        raise Exception()

    def test_header(self):
        self.header = read_edf_header(args.infile)

    def test_channels(self):

        # Obtain the reference channels
        self.ref_channels = PD.read_csv(args.channel_file).values

        # Obtain raw channel names
        raw_channels = [ival['label'] for ival in self.header['SignalHeaders']]

        # Clean up the channel names
        CC            = channel_clean()
        self.channels = CC.direct_inputs(raw_channels)

        # Make sure we have at least the user required channels in the data
        channel_check = []
        for ichannel in self.ref_channels:
            if ichannel in self.channels:
                channel_check.append(True)
            else:
                channel_check.append(False)

        # Make sure all channels are present
        if not all(channel_check):
            raise Exception()

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--infile", type=str, help='Input EDF Filepath')
    parser.add_argument("--sampfreq", type=int, default=512, help='Expected sampling frequency')
    parser.add_argument("--channel_file", type=str, default='configs/hup_standard.csv', help='CSV file containing the expected channels')
    args = parser.parse_args()

    # Run machine level tests
    ML = machine_level(args)
    ML.run_tests()