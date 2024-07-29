import argparse
from sys import exit
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

        # Obtain raw channel names
        raw_channels = [ival['label'] for ival in self.header['SignalHeaders']]

        # Clean up the channel names
        CC            = channel_clean()
        self.channels = CC.direct_inputs(raw_channels)

        #for ikey in 
        print(self.channels)

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