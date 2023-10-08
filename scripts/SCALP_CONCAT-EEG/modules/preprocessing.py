import os
import sys
import inspect
import numpy as np
from tqdm import tqdm
from fractions import Fraction
from scipy.signal import resample_poly, butter, filtfilt

# Local imports
from .yaml_loader import *

class signal_processing:
    
    def __init__(self, data, fs):
        self.data = data
        self.fs   = fs
    
    def butterworth_filter(self, freq_filter_array, filter_type='bandpass', butterorder=3):
        """
        Adopted from Akash Pattnaik code in CNT Research tools.

        Parameters
        ----------
        freq_filter_array : array of integers
            Array of endpoints for frequency filter
        fs : integer
            Sampling frequency.
        filter_type : string, optional, default='bandpass'
            Type of filter to apply. [bandpass,bandstop,lowpass,highpass]
        butterorder: integer, optional, default=3
            Order of the butterworth filter.

        Returns
        -------
            Returns the filtered data.

        """
        
        if filter_type in ["bandpass","bandstop"]:
            bandpass_b, bandpass_a = butter(butterorder,freq_filter_array, btype=filter_type, fs=self.fs)
        elif filter_type in ["lowpass","highpass"]:
            bandpass_b, bandpass_a = butter(butterorder,freq_filter_array[0], btype=filter_type, fs=self.fs)
            
        return filtfilt(bandpass_b, bandpass_a, self.data, axis=0)

    def frequency_downsample(self,output_hz,input_hz=None):
        """
        Adopted from Akash Pattnaik code in CNT Research tools.

        Parameters
        ----------
        output_hz : Integer
            Output dataset frequency.
        input_hz : Integer, optional
            Input frequency. If None, convert all input sampling frequencies to output. If provided, only downsample frequencies that match this value.
            
        Returns
        -------
        Creates new downsampled dataset in instance.

        """

        if input_hz == None and self.fs != output_hz:
            frac                 = Fraction(output_hz, int(self.fs))
            return resample_poly(self.data, up=frac.numerator, down=frac.denominator)
        elif input_hz != None and input_hz == self.fs:
            frac                 = Fraction(output_hz, int(self.fs))
            return resample_poly(self.data, up=frac.numerator, down=frac.denominator)
        else:
            return self.data

class noise_reduction:
    
    def __init__(self, data, fs):
        self.data = data
        self.fs   = fs
    
    def z_score_rejection(self, window_size, z_threshold=5, method="interp"):
        """
        Reject outliers based on the Chebychev theorem. Defaults to <95%/5-sigma.

        Parameters
        ----------
            window_size : integer
                Number of data points before/after current sample to calculate mean/stdev over. Must be odd and 3+. (rounds down if even)
            z_threshold : int, optional
                Number of standard deviation for threshold. Defaults to 5.
            method : str, optional
                Whether to 'mask' (i.e. set to NaN) or 'interp' (i.e. Interpolate over) bad data. Defaults to "interp".
        
        Returns
        -------
        Updates data object in instance.
        """

        # Check parity of window size
        if window_size < 3:
            window_size = 3
        elif window_size%2 == 0:
            window_size -= 1
        
        # Calculate the z values based on sliding window +/- window_size from data point
        pad_size = int(window_size/2)
        pad_data = np.pad(self.data,(pad_size,pad_size), mode='constant', constant_values=np.nan)

        # Calculate the Z-score
        strided_data = np.lib.stride_tricks.sliding_window_view(pad_data, (window_size,))
        stride_inds  = ~np.isnan(strided_data)
        mean         = np.mean(strided_data, axis=1, where=stride_inds)
        variance     = np.mean((strided_data - mean[:, np.newaxis]) ** 2, axis=1, where=stride_inds)
        stdev        = np.sqrt(variance)
        z_vals       = np.zeros(mean.shape)
        inds         = (stdev>0)
        z_vals[inds] = np.fabs(self.data[inds]-mean[inds])/stdev[inds]

        # Replace values   
        mask = (z_vals>=z_threshold)
        if method=="mask" and any(mask):
            self.data[mask] = np.nan
        elif method=="interp" and any(mask):
            x_vals          = np.arange(self.data.size)
            x_vals_interp   = x_vals[~mask]
            y_vals_interp   = np.interp(x_vals,x_vals_interp,self.data[~mask])
            self.data[mask] = y_vals_interp[mask]
        return self.data

class preprocessing:
    """
    This class invokes the various preprocessing steps

    New preprocessing tasks should go into other classes in this script.

    Functions should return the new vector array for each channel/montage channel to be propagated forward.
    """
    
    def __init__(self):
        """
        Use the preprocessing configuration file to step through the preprocessing pipeline on each data array
        in the output data container.
        """
        
        # Read in the preprocessing configuration
        YL = yaml_loader(self.args.preprocess_file)
        config,self.preprocess_commands = YL.return_handler()

        # Get the current module (i.e., the script itself)
        current_module = sys.modules[__name__]

        # Use the inspect module to get a list of classes in the current module
        classes = [cls for name, cls in inspect.getmembers(current_module, inspect.isclass)]

        # Iterate over steps, find the corresponding function, then invoke it.
        steps = np.sort(list(self.preprocess_commands.keys()))
        desc  = "Preprocessing with id %s:" %(self.unique_id)
        for istep in tqdm(steps, desc=desc, total=len(steps), bar_format=self.bar_frmt, position=self.worker_number, leave=False, disable=self.args.silent):

            # Get information about the method
            method_name = self.preprocess_commands[istep]['method']
            method_args = self.preprocess_commands[istep]['args']

            # Clean up any optional arguments set to a null input
            for key, value in method_args.items():
                if type(value) == str:
                    if value.lower() in ['','none']:
                        method_args[key]=None

            for cls in classes:
                if hasattr(cls,method_name):

                    # Loop over the datasets and the channels in each
                    for idx,dataset in enumerate(self.output_list):
                        
                        # Get the input frequencies
                        fs = self.metadata[idx]['fs']

                        # Loop over the channels and get the updated values
                        output = [] 
                        for ichannel in range(dataset.shape[1]):

                            # Perform preprocessing step
                            namespace           = cls(dataset[:,ichannel],fs[ichannel])
                            method_call         = getattr(namespace,method_name)
                            output.append(method_call(**method_args))

                            # Store the new frequencies if downsampling
                            if method_name == 'frequency_downsample':
                                input_fs  = method_args['input_hz']
                                output_fs = method_args['output_hz']
                                if input_fs == None or input_fs == output_fs:
                                    self.metadata[idx]['fs'][ichannel] = output_fs

                        dataset = np.column_stack(output)

                        # Update the data visible by the parent class
                        self.output_list[idx] = dataset

