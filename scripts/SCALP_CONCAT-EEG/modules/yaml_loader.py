import ast
import yaml
import numpy as np

class yaml_loader:

    def __init__(self,yaml_file):
        
        # Read in and typecast the yaml file
        config = yaml.safe_load(open(yaml_file,'r'))
        for ikey in list(config.keys()):
            for jkey in config[ikey]:
                self.str_handler(config[ikey][jkey])
        self.yaml_config = config

        # Make the step sorted dictionary
        self.convert_to_step()

    def return_handler(self):
        return self.yaml_config,self.yaml_step

    def str_handler(self,values):

        if isinstance(values[0], str):
            for idx,ivalue in enumerate(values):
                if ivalue.lower() == '-np.inf':
                    values[idx] = -np.inf
                elif ivalue.lower() == 'np.inf':
                    values[idx] = np.inf
                elif ivalue.lower() == 'none' or ivalue.lower() == '':
                    values[idx] = None
                elif ivalue[0] == '[':
                    values[idx] = ast.literal_eval(ivalue)

    def convert_to_step(self):
        
        # Convert human readable config to easier format for code
        self.yaml_step = {}
        for ikey in list(self.yaml_config.keys()):
            steps = self.yaml_config[ikey]['step_nums']
            for idx,istep in enumerate(steps):

                # Get the argument list for the current command
                args = self.yaml_config[ikey].copy()
                args.pop('step_nums')
                try:
                    args.pop('multithread')
                except KeyError:
                    pass

                # Clean up the current argument list to only show current step
                for jkey in list(args.keys()):
                    args[jkey] = args[jkey][idx]

                # Make the step formatted command list
                self.yaml_step[istep] = {}
                self.yaml_step[istep]['method'] = ikey
                self.yaml_step[istep]['args']   = args

