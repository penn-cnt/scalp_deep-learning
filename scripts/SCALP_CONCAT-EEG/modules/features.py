import sys
import yaml
import inspect
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly, butter, filtfilt

class signal_processing:
    
    def __init__(self, data, fs):
        self.data = data
        self.fs   = fs
    
    def spectral_energy_fooof(self, lo=1, hi=120, relative=True, win_size=2, win_stride=1):

        # bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 80)}
        bands = {"broad":(1,100)}

        # Get the number of samples in each window for welch average and the overlap
        nperseg = int(win_size * fs)
        noverlap = int(win_stride * fs)

        # Calculate the welch periodogram
        freq, pxx = welch(x=x, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1)

        # Initialize a FOOOF object
        fg = FOOOFGroup()

        sys.exit()

        # Set the frequency range to fit the model
        freq_range = [lo, hi]

        # Report: fit the model, print the resulting parameters, and plot the reconstruction
        fg.fit(freq, pxx, freq_range)
        fres = fg.get_results()

        def one_over_f(f, b0, b1):
            return b0 - np.log10(f ** b1)

        idx = np.logical_and(freq >= lo, freq <= hi)
        one_over_f_curves = np.array([one_over_f(freq[idx], *i.aperiodic_params) for i in fres])

        residual = np.log10(pxx[:, idx]) - one_over_f_curves
        freq = freq[idx]

        bandpowers = np.zeros((len(bands), pxx.shape[0]))
        for i_band, (lo, hi) in enumerate(bands.values()):
            if np.logical_and(60 >= lo, 60 <= hi):
                idx1 = np.logical_and(freq >= lo, freq <= 55)
                idx2 = np.logical_and(freq >= 65, freq <= hi)
                bp1 = simpson(
                    y=residual[:, idx1],
                    x=freq[idx1],
                    dx=freq[1] - freq[0]
                )
                bp2 = simpson(
                    y=residual[:, idx2],
                    x=freq[idx2],
                    dx=freq[1] - freq[0]
                )
                bandpowers[i_band] = bp1 + bp2
            else:
                idx = np.logical_and(freq >= lo, freq <= hi)
                bandpowers[i_band] = simpson(
                    y=residual[:, idx],
                    x=freq[idx],
                    dx=freq[1] - freq[0]
                )
        return bandpowers.T


    def spectral_energy_welch(self, low_freq=-np.inf, hi_freq=np.inf, nperseg=None, nseg=10):
        
        # Get the nperseg as needed
        if nperseg == None:
            nseg = int(self.data.shape/nseg)

        # Calculate the Welch periodogram with Hann window (scipy default)
        frequencies, psd = welch(self.data, fs=self.fs, nperseg=256)

        # Find the power in the frequency band
        mask = (frequencies >= low_freq) & (frequencies <= hi_freq)
        return np.trapz(psd[mask], frequencies[mask])

class features:
    
    def __init__(self):
        """
        Use the feature extraction configuration file to step through the preprocessing pipeline on each data array
        in the output data container.
        """
        
        # Read in the preprocessing configuration
        config = yaml.safe_load(open(self.args.feature_file,'r'))
        
        # Convert human readable config to easier format for code
        self.feature_commands = {}
        for ikey in list(config.keys()):
            steps = config[ikey]['step_nums']
            for idx,istep in enumerate(steps):

                # Get the argument list for the current command
                args = config[ikey].copy()
                args.pop('step_nums')
                args.pop('multithread')

                # Clean up the current argument list to only show current step
                for jkey in list(args.keys()):
                    args[jkey] = args[jkey][idx]

                # Make the step formatted command list
                self.feature_commands[istep] = {}
                self.feature_commands[istep]['method'] = ikey
                self.feature_commands[istep]['args']   = args

        # Get the current module (i.e., the script itself)
        current_module = sys.modules[__name__]

        # Use the inspect module to get a list of classes in the current module
        classes = [cls for name, cls in inspect.getmembers(current_module, inspect.isclass)]

        # Iterate over steps, find the corresponding function, then invoke it.
        steps = np.sort(list(self.feature_commands.keys()))
        for istep in steps:

            # Get information about the method
            method_name = self.feature_commands[istep]['method']
            method_args = self.feature_commands[istep]['args']

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
                        fs = next(iter(self.output_meta[idx].values()))['fs']

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
                                    key = list(self.output_meta[idx].keys())[0]
                                    self.output_meta[idx][key]['fs'][ichannel] = output_fs

                        dataset = np.column_stack(output)

                        # Update the data visible by the parent class
                        self.output_list[idx] = dataset

