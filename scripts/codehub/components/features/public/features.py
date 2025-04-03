import os
import ast
import sys
import mne
import yasa
import inspect
import warnings
import numpy as np
import pandas as PD
from tqdm import tqdm
from fooof import FOOOF
from functools import wraps
from scipy.stats import mode
from scipy.integrate import simpson
from sklearn.linear_model import LinearRegression
from scipy.signal import welch, find_peaks, detrend
from neurodsp.spectral import compute_spectrum_welch

# Import error logging (primarily for mne)
from components.core.internal.config_loader import *
from components.metadata.public.metadata_handler import *

# In some cases, we want variables to persist through steps. (i.e. A solver, fitting class, disk i/o, etc.) Persistance_dict can store results across steps.
global persistance_dict
persistance_dict = {}

# Ignore FutureWarnings. Pandas is giving a warning for concat. But the data is not zero. Might be due to a single channel of all NaNs.
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=InconsistentVersionWarning)

def channel_wrapper(method):
    """Decorator to apply a method to each column unless the user directly passes the column to analyze."""
    @wraps(method)
    def wrapper(self, channel=None, *args, **kwargs):
        if channel is not None:
            # Run method on a specific column
            return [method(self, channel, *args, **kwargs)]
        else:
            # Run method on all columns and return results as a dictionary
            return [method(self, col, *args, **kwargs) for col in self.data.columns]
    return wrapper

class channel_wise_metrics:

    def __init__ (self, data, fs, file, fidx, channels=None, trace=False):
        # Manage data typing and form a dataframe as needed
        if isinstance(data, np.ndarray):
            if isinstance(channels,list) or isinstance(channels,np.ndarray):
                if isinstance(channels,np.ndarray):
                    if channels.ndim == 1:
                        channels = list(channels)
                    else:
                        raise ValueError("If passing a numpy arra as channel names, channels must be a 1-d array.")
                self.data     = data
                self.channels = channels
            else:
                raise ValueError("Channels must not be None and be a list or a 1-d array if passing data as a numpy array.")
        elif isinstance(data,PD.DataFrame):
            self.data     = data.values
            self.channels = list(data.columns)

        # Save remaining keywords
        self.fs       = fs
        self.file     = file
        self.fidx     = fidx
        self.trace    = trace
        
        # Because we are using the unmontaged data to infer this feature, we want to map the results to the output channel mapping
        self.outchannel = channels

    def check_persistance(self):
        
        self.channelwise_key = f"channelwise_{self.file}_{self.fidx}_{self.window_length}"
        if self.channelwise_key not in persistance_dict.keys():
            persistance_dict[self.channelwise_key] = {}
            self.fit_ar_model()
            self.calculate_source_sink_space()
        else:
            self.Avg_A = persistance_dict[self.channelwise_key]['Avg_A']
            self.rr    = persistance_dict[self.channelwise_key]['rr']
            self.cr    = persistance_dict[self.channelwise_key]['cr']

    def source_index(self,window_length=0.5):
        """
        Calculate the source index for each channel.

        Args:
            window_length (float, optional): Window length for auto regression in seconds. Defaults to 0.5.
        """

        # Save the window length to the class instance
        self.window_length = window_length

        # Check for any pre-calculated metrics
        self.check_persistance()

        # Get the sink index
        source_index = self.source_fnc()

        # Make the optional tag
        optional_str = f"windowlength_{self.window_length}"

        # Make the output results
        results = [(i_index,optional_str) for i_index in source_index]

        return results

    def sink_index(self,window_length=0.5):
        """
        Calculate the sink index for each channel.

        Args:
            window_length (float, optional): Window length for auto regression in seconds. Defaults to 0.5.
        """

        # Save the window length to the class instance
        self.window_length = window_length

        # Check for any pre-calculated metrics
        self.check_persistance()

        # Get the sink index
        sink_index = self.sink_fnc()

        # Make the optional tag
        optional_str = f"windowlength_{self.window_length}"

        # Make the output results
        results = [(i_index,optional_str) for i_index in sink_index]

        return results

    def fit_ar_model(self):

        # Get window properties
        window_size = self.window_length*self.fs
        n_windows   = self.data.shape[0] // window_size
        
        # Initialize the model insance
        model = LinearRegression()
        
        # Get the auto-regression across windows
        A_all = []
        for i in range(int(n_windows)):

            # Get the current window data
            window_data = self.data[int(i * window_size):int((i + 1) * window_size), :]

            # Get the input vectors
            X = window_data[:-1]

            # Get the output vectors
            y = window_data[1:]

            # Fit the model
            model.fit(X, y)
            
            # Get the slope and store the running list
            A = model.coef_
            A_all.append(A)
        
        # Get the absolute mean of all linear slopes
        A_all = np.array(A_all)
        Avg_A = np.mean(A_all, axis=0)
        Avg_A = np.abs(Avg_A)

        # Store result to class instance
        self.Avg_A = Avg_A

        # Store to persistance dict
        persistance_dict[self.channelwise_key]['Avg_A'] = self.Avg_A

    def calculate_source_sink_space(self):
        
        # Update the main diagonal to avoid self-same calculatuon
        np.fill_diagonal(self.Avg_A, 0)

        # Get the node strength for column and row-wise
        i_node_strength = np.sum(self.Avg_A, axis=0)
        j_node_strength = np.sum(self.Avg_A, axis=1)

        # rank node strengths to get row rank (rr) and column rank (cr)
        self.rr = self.calculate_rank(i_node_strength)
        self.cr = self.calculate_rank(j_node_strength)

        # Store to persistance dict
        persistance_dict[self.channelwise_key]['rr'] = self.rr
        persistance_dict[self.channelwise_key]['cr'] = self.cr

    def calculate_rank(self,arr): 

        # Get the indices that would sort the array in descending order
        sorted_indices = np.argsort(arr)[::-1]

        # Create a ranking array
        ranks                 = np.zeros_like(arr, dtype=float)
        ranks[sorted_indices] = np.arange(1, len(arr) + 1) / len(ranks)

        return ranks

    def source_fnc(self):

        # Calculate the source index
        N             = len(self.rr)
        x             = self.rr - (1/N)
        y             = self.cr - 1
        vector_length = np.sqrt(x**2 + y**2)
        source_index  = np.sqrt(2) - vector_length
        return source_index

    def sink_fnc(self):

        # Calculate source sink
        N             = len(self.rr)
        x             = self.rr - 1
        y             = self.cr - (1/N)
        vector_length = np.sqrt(x**2 + y**2)
        sink_index    = np.sqrt(2) - vector_length
        return sink_index

class YASA_processing:
    """
    Yasa sleep staging feature extraction.
    """
    
    def __init__ (self, data, fs, channels=None, trace=False):
        # Manage data typing and form a dataframe as needed
        if isinstance(data, np.ndarray):
            if isinstance(channels,list) or isinstance(channels,np.ndarray):
                if isinstance(channels,np.ndarray):
                    if channels.ndim == 1:
                        channels = list(channels)
                    else:
                        raise ValueError("If passing a numpy arra as channel names, channels must be a 1-d array.")
                self.data     = data
                self.channels = channels
            else:
                raise ValueError("Channels must not be None and be a list or a 1-d array if passing data as a numpy array.")
        elif isinstance(data,PD.DataFrame):
            self.data     = data.values
            self.channels = list(data.columns)

        # Save remaining keywords
        self.fs       = fs
        self.trace    = trace
        
        # Because we are using the unmontaged data to infer this feature, we want to map the results to the output channel mapping
        self.outchannel = channels

    def make_montage_object(self,config_path):

        # Create the mne channel types
        fp      = open(config_path,'r')
        mapping = yaml.safe_load(fp)
        fp.close()
        persistance_dict['mne_mapping'] = mapping

    def make_raw_object(self,config_path):

        # Get the channel mappings in mne compliant form
        if 'mne_mapping' not in persistance_dict.keys():
            self.make_montage_object(config_path)
        mapping      = persistance_dict['mne_mapping']
        mapping_keys = list(mapping.keys())

        # Assign the mapping to each channel
        ch_types     = []
        for ichannel in self.channels:
            if ichannel in mapping_keys:
                ch_types.append(mapping[ichannel])
            else:
                ch_types.append('eeg')

        # Create the raw mne object and set the reference
        info     = mne.create_info(self.channels, self.fs, ch_types=ch_types,verbose=False)
        self.raw = mne.io.RawArray(self.data.T, info, verbose=False)

    def yasa_sleep_stage(self,config_path,consensus_channels=['CZ','C03','C04']):

        # Check for a long enough duration
        if (self.data.shape[0]/self.fs/60 >=5):
            # Make the raw object for YASA to work with
            self.make_raw_object(config_path)

            # Set the right reference for eyeblink removal (CAR by default)
            self.raw = self.raw.set_eeg_reference('average',verbose=False)

            # Apply the minimum needed filter for eyeblink removal
            self.raw = self.raw.filter(0.5,30,verbose=False)

            # Resample down to 100 HZ
            self.raw = self.raw.resample(100)

            # Get the yasa prediction
            results = []
            for ichannel in consensus_channels:
                sls = yasa.SleepStaging(self.raw, eeg_name=ichannel)
                results.append(list(sls.predict()))
            results = np.array(results)

            # Get the epipy formatted output
            output = ''
            for irow in results.T:
                output += ','.join(irow)
                output += '|'
            output = output[:-1]
        else:
            output = None
        
        # Make the optional string. In this case, the consensus channel list
        optional_str = ','.join(consensus_channels)
        
        # Reformat the output to match the output structure
        results = [(output,optional_str) for ichannel in self.outchannel]

        return results
 
class FOOOF_processing:

    def __init__(self, data, fs, freq_range, file, fidx, ichannel, trace=False):
        """
        Initalize the FoooF code. If directly invoking, you can set file, fidx, and ichannel to any dummy values.

        Args:
            data (_type_): _description_
            fs (_type_): _description_
            freq_range (_type_): _description_
            file (_type_): _description_
            fidx (_type_): _description_
            ichannel (_type_): _description_
            trace (bool, optional): Return the traced vectors from the bandpower measurement. Defaults to False.
        """

        self.data       = data
        self.fs         = fs
        self.freq_range = freq_range
        self.file       = file
        self.fidx       = fidx
        self.ichannel   = ichannel
        self.trace      = trace

    def direct_invocation(self,lo_freq,hi_freq,win_size=2,win_stride=1):
        self.create_initial_power_spectra()
        self.fit_fooof()
        b0,_      = self.aperiodic_b0(win_size,win_stride)
        b1,_      = self.aperiodic_b1(win_size,win_stride)
        fooof_psd = self.fooof_bandpower(lo_freq,hi_freq, win_size, win_stride)
        return b0,b1,fooof_psd

    def create_initial_power_spectra(self):
        self.freqs, initial_power_spectrum = welch(x=self.data.reshape((-1,1)), fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap, axis=0)
        initial_power_spectrum             = initial_power_spectrum.flatten()
        inds                               = (self.freqs>=0.5)&np.isfinite(initial_power_spectrum)&(initial_power_spectrum>0)
        freqs                              = self.freqs[inds]
        initial_power_spectrum             = initial_power_spectrum[inds]
        self.initial_power_spectrum        = np.interp(self.freqs,freqs,initial_power_spectrum)

    def fit_fooof(self):

        # Initialize a FOOOF object
        fg = FOOOF(peak_width_limits=(2,12))

        # Report: fit the model, print the resulting parameters, and plot the reconstruction
        if self.freqs.size > 0:
            fg.fit(self.freqs, self.initial_power_spectrum, self.freq_range)

            # get the one over f curve
            b0,b1 = fg.get_results().aperiodic_params
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                one_over_f = b0-np.log10(self.freqs**b1)

            # Get the residual periodic fit
            periodic_comp = self.initial_power_spectrum-one_over_f

            # Store the results for persistant fitting
            persistance_dict[self.fooof_key] = (fg,self.freqs,periodic_comp,self.initial_power_spectrum)
        else:
            persistance_dict[self.fooof_key] = (None,None,None)

    def check_persistance(self):
        
        self.fooof_key = f"fooof_{self.file}_{self.fidx}_{self.ichannel}"
        if self.fooof_key not in persistance_dict.keys():
            self.create_initial_power_spectra()
            self.fit_fooof()

    def fooof_aperiodic_b0(self, win_size=2., win_stride=1.):
        """
        Return the constant offset for the aperiodic fit.

        Returns:
            float: Unitless float for the aperiodic offset parameter.
        """

        # Make the optional tag to identify the dataslice
        self.optional_tag = ''

        # Get the number of samples in each window for welch average and the overlap
        self.nperseg  = int(float(win_size) * self.fs)
        self.noverlap = int(float(win_stride) * self.fs)

        # Check for fooof model
        self.check_persistance()

        # get the needed object from the persistance dictionary
        fg = persistance_dict[self.fooof_key][0]
        if fg != None:
            b0 = fg.aperiodic_params_[0]
        else:
            b0 = None

        return b0,self.optional_tag
    
    def fooof_aperiodic_b1(self, win_size=2., win_stride=1.):
        """
        Return the powerlaw exponent for the aperiodic fit.

        Returns:
            float: Unitless float for the aperiodic powerlaw parameter.
        """

        # Make the optional tag to identify the dataslice
        self.optional_tag = ''

        # Get the number of samples in each window for welch average and the overlap
        self.nperseg  = int(float(win_size) * self.fs)
        self.noverlap = int(float(win_stride) * self.fs)

        # Check for fooof model
        self.check_persistance()

        # get the needed object from the persistance dictionary
        fg = persistance_dict[self.fooof_key][0]
        if fg != None:
            b1 = fg.aperiodic_params_[1]
        else:
            b1 = None

        return b1,self.optional_tag

    def fooof_bandpower(self,lo_freq,hi_freq, win_size=2., win_stride=1.):
        """
        Return the bandpower with the aperiodic component removed.

        Returns:
            float: Bandpower in the periodic component in the given frequency band.
        """

        # Add in the optional tagging to denote frequency range of this step
        low_freq_str      = f"{lo_freq:.2f}"
        hi_freq_str       = f"{hi_freq:.2f}"
        self.optional_tag = '['+low_freq_str+','+hi_freq_str+']'

        # Get the number of samples in each window for welch average and the overlap
        self.nperseg  = int(float(win_size) * self.fs)
        self.noverlap = int(float(win_stride) * self.fs)

        # Check for fooof model
        self.check_persistance()

        # Get the needed object, then retrieve data
        x    = persistance_dict[self.fooof_key][1]
        y    = persistance_dict[self.fooof_key][2]
        rawy = persistance_dict[self.fooof_key][3]

        # Get the correct array slice to return the simpson integration
        if isinstance(x, np.ndarray):
            inds = (x>=lo_freq)&(x<hi_freq)
            intg = simpson(y=y[inds],x=x[inds])
        else:
            intg = None

        if not self.trace:
            return intg,self.optional_tag
        else:
            return intg,self.optional_tag,(['freqs','psd_fooof','psd_welch'],x.astype('float16'),y.astype('float32'),rawy.astype('float32'))

class signal_processing:
    """
    Class devoted to basic signal processing tasks. (Band-power/peak-finder/etc.)

    Uses only one channel at a time.
    """
    
    def __init__(self, data, fs, channels=None, trace=False):
        """
        Store the dataframe object to the signal processing class for use in different methods.

        Args:
            data (array or dataframe): Array/DataFrame of timeseries data. Row=Sample, Column=Channel.
            fs (float): Sampling Frequency
            channels (list): List of channel names. Same order as columns in data.
            trace (bool, optional): _description_. Defaults to False.
        """

        # Manage data typing and form a dataframe as needed
        if isinstance(data, np.ndarray):
            if isinstance(channels,list) or isinstance(channels,np.ndarray):
                if isinstance(channels,np.ndarray):
                    if channels.ndim == 1:
                        channels = list(channels)
                    else:
                        raise ValueError("If passing a numpy arra as channel names, channels must be a 1-d array.")
                self.data     = data
                self.channels = channels
            else:
                raise ValueError("Channels must not be None and be a list or a 1-d array if passing data as a numpy array.")
        elif isinstance(data,PD.DataFrame):
            print("Shallow copy")
            self.data     = data.values
            self.channels = list(data.columns)

        # Save remaining keywords
        self.fs       = fs
        self.trace    = trace
    
    @channel_wrapper
    def spectral_energy_welch(self, channel, low_freq=-np.inf, hi_freq=np.inf, win_size=2., win_stride=1.):
        """
        Returns the spectral energy using the Welch method.

        Args:
            low_freq (float, optional): Low frequency cutoff. Defaults to -np.inf.
            hi_freq (float, optional): High frequency cutoff. Defaults to np.inf.
            win_size (float, optional): Window size in units of sampling frequency. Defaults to 2.
            win_stride (float, optional): Window overlap in units of sampling frequency. Defaults to 1.

        Returns:
            spectral_energy (float): Spectral energy within the frequency band.
            optional_tag (string): Unique identifier that is added to the output dataframe to show the frequency window for which a welch spectral energy was calculated.
            (frequencies,psd): If trace is enabled for this pipeline, return the frequencies and psd for this channel for testing.
        """

        # Add in the optional tagging to denote frequency range of this step
        low_freq_str      = f"{low_freq:.2f}"
        hi_freq_str       = f"{hi_freq:.2f}"
        self.optional_tag = '['+low_freq_str+','+hi_freq_str+']'

        # Get the number of samples in each window for welch average and the overlap
        nperseg = int(float(win_size) * self.fs)
        noverlap = int(float(win_stride) * self.fs)

        # Calculate the welch periodogram
        idata                               = self.data[channel].values
        frequencies, initial_power_spectrum = welch(x=idata.reshape((-1,1)), fs=self.fs, nperseg=nperseg, noverlap=noverlap, axis=0)
        initial_power_spectrum              = initial_power_spectrum.flatten()
        inds                                = (frequencies>=0.5)&np.isfinite(initial_power_spectrum)&(initial_power_spectrum>0)
        freqs                               = frequencies[inds]
        initial_power_spectrum              = initial_power_spectrum[inds]
        psd                                 = np.interp(frequencies,freqs,initial_power_spectrum)

        # Calculate the spectral energy
        mask            = (frequencies >= low_freq) & (frequencies <= hi_freq)
        spectral_energy = np.trapz(psd[mask], frequencies[mask])

        if not self.trace:
            return spectral_energy,self.optional_tag
        else:
            return spectral_energy,self.optional_tag,(['freqs','psd_welch'],frequencies.astype('float16'),psd.astype('float32'))

    @channel_wrapper
    def normalized_spectral_energy_welch(self, channel, low_freq=-np.inf, hi_freq=np.inf, win_size=2., win_stride=1.):
        """
        Returns the spectral energy using the Welch method.

        Args:
            low_freq (float, optional): Low frequency cutoff. Defaults to -np.inf.
            hi_freq (float, optional): High frequency cutoff. Defaults to np.inf.
            win_size (float, optional): Window size in units of sampling frequency. Defaults to 2.
            win_stride (float, optional): Window overlap in units of sampling frequency. Defaults to 1.

        Returns:
            spectral_energy (float): Spectral energy within the frequency band.
            optional_tag (string): Unique identifier that is added to the output dataframe to show the frequency window for which a welch spectral energy was calculated.
            (frequencies,psd): If trace is enabled for this pipeline, return the frequencies and psd for this channel for testing.
        """

        # Add in the optional tagging to denote frequency range of this step
        low_freq_str      = f"{low_freq:.2f}"
        hi_freq_str       = f"{hi_freq:.2f}"
        self.optional_tag = '['+low_freq_str+','+hi_freq_str+']'

        # Get the number of samples in each window for welch average and the overlap
        nperseg = int(float(win_size) * self.fs)
        noverlap = int(float(win_stride) * self.fs)

        # Calculate the welch periodogram
        idata                               = self.data[channel].values
        frequencies, initial_power_spectrum = welch(x=idata.reshape((-1,1)), fs=self.fs, nperseg=nperseg, noverlap=noverlap, axis=0)
        initial_power_spectrum              = initial_power_spectrum.flatten()
        inds                                = (frequencies>=0.5)&np.isfinite(initial_power_spectrum)&(initial_power_spectrum>0)
        freqs                               = frequencies[inds]
        initial_power_spectrum              = initial_power_spectrum[inds]
        psd                                 = np.interp(frequencies,freqs,initial_power_spectrum)

        # Calculate the spectral energy
        mask            = (frequencies >= low_freq) & (frequencies <= hi_freq)
        spectral_energy = np.trapz(psd[mask], frequencies[mask])/np.trapz(psd, frequencies)

        if not self.trace:
            return spectral_energy,self.optional_tag
        else:
            return spectral_energy,self.optional_tag,(['freqs','psd_welch'],frequencies.astype('float16'),psd.astype('float32'))

    @channel_wrapper
    def topographic_peaks(self,channel,prominence_height,min_width,height_unit='zscore',width_unit='seconds',detrend_flag=False):
        """
        Find the topographic peaks in channel data. This is a naive/fast way of finding spikes or slowing.

        Args:
            prominence_height (_type_): Prominence threshold.
            min_width (_type_): Minimum width of peak.
            height_unit (str, optional): Unit for prominence height. zscore=Height by zscore. Else, absolute height. Defaults to 'zscore'.
            width_unit (str, optional): Unit for width of peaks. 'seconds'=width in seconds. Else, width in bins. Defaults to 'seconds'.
            detrend_flag (bool, optional): Detrend the data before searching for peaks. Defaults to False.

        Returns:
            out (string): Underscore concatenated string of peak, left edge of peak, and right edge of peak
            optional_tag (string): Underscore concatenated string of prominence height, width, and their unit types for this feature step to unique identify the feature.
        """

        # Make the optional tag output
        self.optional_tag = f"{prominence_height:.2f}_{height_unit}_{min_width:.2f}_{width_unit}_{detrend_flag}"

        # Detrend as needed
        if detrend_flag:
            data = detrend(self.data[channel].values)
        else:
            data = np.copy(self.data[channel].values)

        # Recast height into a pure number as needed
        if height_unit == 'zscore':
            prominence_height_input = np.median(data)+prominence_height*np.std(data)
        else:
            prominence_height_input = prominence_height

        # Recast width into a pure number as needed
        if width_unit == 'seconds':
            min_width_input = min_width*self.fs
        else:
            min_width_input = min_width

        # Calculate the peak info
        output = find_peaks(data,prominence=prominence_height_input,width=min_width_input)

        # Get peak info
        try:
            peak       = output[0][0]
            lwidth     = peak-output[1]['left_ips'][0]
            rwidth     = output[1]['right_ips'][0]-peak
        except IndexError:
            peak   = None
            lwidth = None
            rwidth = None

        # We can only return a single object that is readable by pandas, so pack results into a string to be broken down later by user
        out = [peak,lwidth,rwidth]

        # Return a tuple of (peak, left width, right width) to store all of the peak info
        return out,self.optional_tag
    
    @channel_wrapper
    def line_length(self,channel):
        """
        Return the line length along the given channel.

        Returns:
            LL (float): Line length
            optional_tag (string): Optional tag
        """

        LL           = np.sum(np.abs(np.ediff1d(self.data[channel].values)))
        optional_tag = ''
        return LL,optional_tag

class basic_statistics:
    """
    Basic features that can be extracted from the raw time series data.
    """

    def __init__(self, data, fs, channels=None, trace=False):
        """
        Store the dataframe object to the signal processing class for use in different methods.

        Args:
            data (array or dataframe): Array/DataFrame of timeseries data. Row=Sample, Column=Channel.
            fs (float): Sampling Frequency
            channels (list): List of channel names. Same order as columns in data.
            trace (bool, optional): _description_. Defaults to False.
        """
        
        # Manage data typing and form a dataframe as needed
        if isinstance(data, np.ndarray):
            if isinstance(channels,list) or isinstance(channels,np.ndarray):
                if isinstance(channels,np.ndarray):
                    if channels.ndim == 1:
                        channels = list(channels)
                    else:
                        raise ValueError("If passing a numpy arra as channel names, channels must be a 1-d array.")
                self.data     = data
                self.channels = channels
            else:
                raise ValueError("Channels must not be None and be a list or a 1-d array if passing data as a numpy array.")
        elif isinstance(data,PD.DataFrame):
            self.data     = data.values
            self.channels = list(data.columns)

        # Save remaining keywords
        self.fs       = fs
        self.trace    = trace

    @channel_wrapper
    def mean(self,channel):
        """
        Returns the mean value in a channel.

        Returns:
            float: Mean channel intensity.
        """

        return np.mean(self.data[channel].values),'mean'

    @channel_wrapper
    def median(self,channel):
        """
        Returns the median value in a channel.

        Returns:
            float: Median channel intensity.
        """

        return np.median(self.data[channel].values),'median'
    
    @channel_wrapper
    def stdev(self,channel):
        """
        Returns the standard deviation in a channel.

        Returns:
            float: Standard deviation in a channel.
        """

        return np.std(self.data[channel].values),'stdev'
    
    @channel_wrapper
    def quantile(self,channel,q,method='median_unbiased'):
        """
        Returns the q-th quantile of the data.

        Args:
            q (float): Probability to measure the quantile from. q∈[0:1] .
            method (str, optional): Interpolation method. Defaults to 'median_unbiased'.
        """

        optional_tag = f"quantile_{q:.2f}"
        return np.quantile(self.data[channel].values,q=q,method=method),optional_tag
    
    @channel_wrapper
    def rms(self,channel):
        """
        Returns the mean root mean square of the channel.
        """

        val = np.sum(self.data[channel].values**2)/self.data[channel].values.size
        return np.sqrt(val),'rms'

class features:
    """
    This class invokes the various features that can be calculated. This should not be altered without good reason.

    New feature extraction tasks should go into other classes in this script. Each feature should return either the scalar feature value, 
    or a tuple with the scalar and some optional tagging for additional group distinctions. (i.e. A welch bandpower feature would return the 
    bandpower, plus an optional tag denoting what the frequency boundaries were.)

    If you need a vector returned, or need to avoid repeated calculations across different features, you can use the persistance_dict object
    and return the required elements across different feature calls.The FOOOF class has examples of how to use this object.

    As of 06/10/24, the default behavior is to pass a class a vector with the channel data and the sampling frequency. This behavior can be altered
    at the `CLASS INITIALIZATION` code block. 
    """

    def __init__(self):
        """
        Use the feature extraction configuration file to step through the preprocessing pipeline on each data array
        in the output data container.
        """

        # Initialize some variables
        channels  = self.montage_channels.copy()
        outcols   = ['file','t_start','t_end','t_window','method','tag']+channels

        print(channels)
        exit()

        # Read in the feature configuration
        YL = config_loader(self.args.feature_file)
        config,self.feature_commands = YL.return_handler()

        # Get the current module (i.e., the script itself)
        current_module = sys.modules[__name__]

        # Use the inspect module to get a list of classes in the current module
        classes = [cls for name, cls in inspect.getmembers(current_module, inspect.isclass)]

        # Make a dummy list so we can append files to the dataframe in a staggered fashion (performance improvement)
        df_values = []

        # Iterate over steps, find the corresponding function, then invoke it.
        steps = np.sort(list(self.feature_commands.keys()))
        desc  = "Feature extraction with id %s:" %(self.unique_id)
        for istep in tqdm(steps, desc=desc, total=len(steps), bar_format=self.bar_frmt, position=self.worker_number, leave=False, disable=self.args.silent,dynamic_ncols=True):

            # Get information about the method
            method_name = self.feature_commands[istep]['method']
            method_args = self.feature_commands[istep]['args']
            error_flag  = False

            for cls in classes:
                if hasattr(cls,method_name):
                    # Loop over the datasets and the channels in each
                    for idx,dataset in enumerate(self.output_list):
                        
                        # Grab the current meta data object
                        meta_idx = self.output_meta[idx]
                        imeta    = self.metadata[meta_idx]

                        # Get the input frequencies
                        fs = imeta['fs'][0]

                        # Obtain the features
                        output = []
                        try:
                            # Get the input arguments for the current step
                            for key, value in method_args.items():
                                try:
                                    method_args[key] = ast.literal_eval(value)
                                except:
                                    pass

                            # Create namespaces for each class. Then choose which style of initilization is used by logic gate.
                            if cls.__name__ == 'FOOOF_processing':
                                # DEPRECIATED FORMAT! Will not work. Should mirror channel_wise metrics going forward.
                                namespace = cls(idata,fs,[0.5,32], imeta['file'], idx, ichannel, self.args.trace)
                            elif cls.__name__ == 'channel_wise_metrics':
                                namespace = cls(dataset,fs,imeta['file'], idx, channels, self.args.trace)
                            elif cls.__name__ == 'YASA_processing':
                                namespace = cls(imeta['unmontaged_data'],fs,channels,self.args.trace)
                            else:
                                namespace = cls(dataset,fs,channels,self.args.trace)

                            # Get the method name and return results from the method
                            method_call = getattr(namespace,method_name)
                            results     = method_call(**method_args)
                            result_a    = [iresult[0] for iresult in results]
                            result_b    = results[0][1]

                            # If the user wants to trace some values (see the results as they are processed), they can return result_c
                            if len(results[0]) == 3:
                                for ii,ichannel in enumerate(channels):
                                    # Get the lower level column labels
                                    cols = results[ii][2][0]
                                    vals = results[ii][2][1:]

                                    # Make the dictionary to nest into metadata
                                    inner_dict = dict(zip(cols,vals))
                                    tracemeta  = {ichannel:inner_dict}

                                    # Add the trace to the metadata
                                    metadata_handler.add_metadata(self,idx,method_name,tracemeta)

                            # Extend the output with results
                            output.extend(result_a)
                        except IndexError:#Exception as e:

                            # Add the ability to see the error if debugging
                            if self.args.debug:
                                fname       = os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1]
                                error_type  = sys.exc_info()[0]
                                line_number = sys.exc_info()[2].tb_lineno
                                print(f"Error type {error_type} in line {line_number} for {method_name}. Error message: {e}")
                                exit()

                            # We need a flexible solution to errors, so just populating a nan value
                            output.extend([None for ii in range(len(channels))])
                            try:
                                result_b = getattr(namespace,'optional_tag')
                            except:
                                result_b = "None"
                            
                            # Save the error for this step
                            if not error_flag and not self.args.debug:
                                error_dir = f"{self.args.outdir}errors/"
                                if not os.path.exists(error_dir):
                                    os.system(f"mkdir -p {error_dir}")

                                fp = open(f"{error_dir}{self.worker_number}_features.error","a")
                                fp.write(f"Step {istep:02}/{method_name}: Error {e}\n")
                                fp.close()
                                error_flag = True

                        # Use metadata to allow proper feature grouping
                        meta_arr = [imeta['file'].split('/')[-1],imeta['t_start'],imeta['t_end'],imeta['t_window'],method_name,result_b]
                        df_values.append(np.concatenate((meta_arr,output),axis=0))

                        # Stagger condition for pandas concat
                        if (idx%5000==0):

                            # Dataframe creations
                            iDF = PD.DataFrame(df_values,columns=outcols)
                            if not iDF[channels].isnull().values.all() and not iDF[channels].isna().values.all():
                                try:
                                    self.feature_df = PD.concat((self.feature_df,iDF))
                                except AttributeError:
                                    self.feature_df = iDF.copy()

                            # Clean up the dummy list
                            df_values = []

                    # Dataframe creations
                    iDF = PD.DataFrame(df_values,columns=outcols)
                    if not iDF[channels].isnull().values.all() and not iDF[channels].isna().values.all():
                        try:
                            self.feature_df = PD.concat((self.feature_df,iDF))
                        except AttributeError:
                            self.feature_df = iDF.copy()

                    # Downcast feature array to take up less space in physical and virtual memory. Use downcast first in case its a feature that cannot be made numeric
                    for ichannel in channels:
                        try:
                            self.feature_df[ichannel]=self.feature_df[ichannel].astype('float32')
                        except ValueError:
                            pass

        # The stagger condition seems to add duplicates. Need to fix eventually.
        self.feature_df = self.feature_df.drop_duplicates(ignore_index=True)