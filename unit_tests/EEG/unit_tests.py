import argparse
import numpy as np
import pandas as PD
from sys import exit
from mne.io import read_raw_edf
from pyedflib.highlevel import read_edf,read_edf_header
from components.workflows.public.channel_clean import channel_clean


class machine_level:

    def __init__(self,args):

        self.args = args

    def run_tests(self):
        
        try:
            self.test_header()
            self.test_channels()
            self.test_sampfreq()
            self.load_data_mne()
            self.load_data_pyedf()
            self.compare_libraries()
            self.check_nan()
            self.check_running_stats(self.args.sampfreq+1)
        except Exception as e:
            print(e)
            exit(1)

    def failure(self):
        raise Exception()

    def test_header(self):
        self.header = read_edf_header(args.infile)

    def test_sampfreq(self):

        # Obtain raw channel names
        samp_freqs = np.array([int(ival['sample_rate']) for ival in self.header['SignalHeaders']])

        # Check against the expected frequency
        freq_mask = (samp_freqs!=self.args.sampfreq)
        if (freq_mask).any():
            raise Exception(f"Unexpted sampling frequency found in {self.channels[freq_mask]}")

    def test_channels(self):

        # Obtain the reference channels
        self.ref_channels = PD.read_csv(args.channel_file,names=['channels']).values

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
        
        # Check number of channels
        if self.channels.size != self.ref_channels.size:
            raise Exception("Did not receive expected number of channels. This can arise due to poorly inputted channels.")

    def load_data_mne(self):
        self.mne_data = read_raw_edf(self.args.infile).get_data()

    def load_data_pyedf(self):
        self.pyedf_data, self.pyedf_chan_info,_ = read_edf(self.args.infile)
        self.pyedf_data *= 1e-6

    def compare_libraries(self,tol=1e-8):

        diffs=self.mne_data-self.pyedf_data
        if (diffs>tol).any():
            exit(1)

        # Drop the pyedf data to reduce memory usage now that we dont need it
        self.pyedf_data = None

    def check_nan(self):

        if np.isnan(self.mne_data).any():
            raise Exception("NaNs found in the data.")
        
    def check_running_stats(self,window_size):

        variance_array = np.zeros(self.mne_data.shape)
        for idx in range(self.channels.size):
            data = self.mne_data[idx]

            # Check parity of window size
            if window_size < 3:
                window_size = 3
            elif window_size%2 == 0:
                window_size -= 1

            # Make a padded data entry
            pad_size = int(window_size/2)
            pad_data = np.pad(data,(pad_size,pad_size), mode='constant', constant_values=np.nan)

            # Calculate the median and variance
            strided_data = np.lib.stride_tricks.sliding_window_view(pad_data, (window_size,))
            stride_inds  = ~np.isnan(strided_data)
            mean         = np.mean(strided_data, axis=1, where=stride_inds)
            variance     = np.mean((strided_data - mean[:, np.newaxis]) ** 2, axis=1, where=stride_inds)

            # Store to the channel wide measurement
            variance_array[idx] = variance

        # Get the channel wide variance sum. Zero means all channels had zero variance for the window size
        mask = variance_array.sum(axis=0)==0
        if (mask).any():
            raise Exception(f"All channels have zero variance around second {self.args.sampfreq*np.arange(mask.size)[mask]} seconds.")

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--infile", type=str, help='Input EDF Filepath')
    parser.add_argument("--sampfreq", type=int, default=256, help='Expected sampling frequency')
    parser.add_argument("--channel_file", type=str, default='configs/hup_standard.csv', help='CSV file containing the expected channels')
    args = parser.parse_args()

    # Run machine level tests
    ML = machine_level(args)
    ML.run_tests()