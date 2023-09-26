# General libraries
import torch
import pickle
import datetime
import numpy as np
import pandas as PD


# Import the classes
from .data_loader import *
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .data_viability import *

class output_manager:

    def __init__(self):

        self.output_list = []
        self.output_meta = []

    def update_output_list(self,data,meta):

        self.output_list.append(data)
        self.output_meta.append(meta)

    def save_output_list(self):

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        pickle.dump(self.output_list,open("%s_data.pickle" %(timestamp),"wb"))
        pickle.dump(self.output_meta,open("%s_meta.pickle" %(timestamp),"wb"))

    def create_tensor(self):

        # Create the tensor
        self.input_tensor_dataset = [torch.utils.data.DataLoader(dataset) for dataset in self.output_list]