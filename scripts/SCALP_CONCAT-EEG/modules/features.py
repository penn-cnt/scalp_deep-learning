import ast
import sys
import yaml
import inspect
import numpy as np
from fooof import FOOOFGroup
from fractions import Fraction
from scipy.signal import welch


class signal_processing:
    
    def __init__(self, data, fs):
        self.data = data
        self.fs   = fs
    
    def spectral_energy_welch(self, low_freq=-np.inf, hi_freq=np.inf, win_size=2, win_stride=1):

        # bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30), "gamma": (30, 80)}

        # Get the number of samples in each window for welch average and the overlap
        nperseg = int(float(win_size) * self.fs)
        noverlap = int(float(win_stride) * self.fs)

        # Calculate the welch periodogram
        frequencies, psd = welch(x=self.data.reshape((-1,1)), fs=self.fs, nperseg=nperseg, noverlap=noverlap, axis=0)

        # Calculate the spectral energy
        print(low_freq)
        print(type(low_freq))

        mask            = (frequencies >= low_freq) & (frequencies <= hi_freq)
        spectral_energy = np.trapz(psd[mask], frequencies[mask])

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
                try:
                    args.pop('multithread')
                except KeyError:
                    pass

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

                            for key, value in method_args.items():
                                try:
                                    method_args[key] = ast.literal_eval(value)
                                except:
                                    pass

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

