import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly, butter, filtfilt

class signal_processing:
    
    def __init__(self, data):
        self.data = data
    
    def butterworth_filter(self, freq_filter_array, fs, filter_type=='bandpass', butterorder=3):
        
        if filter_type in ['bandpass','bandstop']:
            bandpass_b, bandpass_a = butter(order,freq_filter_array, btype=filter_type, fs=fs)
        elif filter_type in [‘lowpass’,‘highpass’]:
            bandpass_b, bandpass_a = butter(order,freq_filter_array[0], btype=filter_type, fs=fs)
            
        return filtfilt(bandpass_b, bandpass_a, self.data, axis=0)

def noise_reduction:
    
    def __init__(self, data):
        self.data = data
    
    def z_score_rejection(self, window_size, z_threshold=5, method="interp"):
        
        # Calculate the z values based on sliding window
        z_vals = []
        for idx,ival in enumerate(self.data):
            lind  = np.max([idx-window_size,0])
            rind  = np.min([idx+window_size,self.data.size-1])
            vals  = self.data[lind:rind]
            mean  = np.mean(vals)
            stdev = np.std(vals)
            z_vals.append(np.fabs(ival-mean)/stdev)
        
        # Replace values       
        mask = (z_vals>=z_threshold)
        if method=="mask":
            self.data[mask] = np.nan
        elif method=="interp":
            x_vals          = np.arange(self.data.size)
            x_vals_interp   = x_vals[~mask]
            y_vals_interp   = np.interp(x_vals,x_vals_interp,self.data[~mask])
            self.data[mask] = y_vals_interp[mask]
        
        
        

class preprocessing:
    
    def __init__(self):
        pass
    
    def frequency_downsample(self,input_hz,output_hz):
        """
        Adopted from Akash Pattnaik code in CNT Research tools.

        Parameters
        ----------
        input_hz : Integer
            Original dataset frequency.
        output_hz : TYPE
            Output dataset frequency.

        Returns
        -------
        New downsampled dataset.

        """

        if input_hz != output_hz:
            frac                 = Fraction(new_fs, int(fs))
            self.proprocess_data = resample_poly(self.proprocess_data, up=frac.numerator, down=frac.denominator)
            
