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

class tensor_manager:

    def __init__(self):

        self.input_tensor_list = []

    def update_tensor_list(self,data):

        self.input_tensor_list.append(data)

    def create_tensor(self,input_array):

        # Create the tensor
        self.input_tensor_dataset = [torch.utils.data.DataLoader(dataset) for dataset in input_array]