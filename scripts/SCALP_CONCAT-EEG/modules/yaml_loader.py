import yaml
import numpy as np

class yaml_loader:

    def __init__(self,yaml_file):
        
        config = yaml.safe_load(open(yaml_file,'r'))
        for ikey in list(config.keys()):
            for jkey in config[ikey]:
                self.type_handler(config[ikey][jkey])

    def type_handler(self,values):

        if isinstance(values[0], str):
            for idx,ivalue in enumerate(values):
                if ivalue.lower() == '-np.inf':
                    values[idx] = -np.inf
                elif ivalue.lower() == 'np.inf':
                    values[idx] = np.inf
                elif ivalue.lower() == 'none':
                    values[idx] = None


