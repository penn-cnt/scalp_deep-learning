# General libraries
import torch
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

    def update_output_list(self,data):

        self.output_list.append(data)

    def create_tensor(self):

        # Create the tensor
        self.input_tensor_dataset = [torch.utils.data.DataLoader(dataset) for dataset in self.output_list]