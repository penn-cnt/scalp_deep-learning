import numpy as np

# Torch loaders
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ray imports
from ray import train, tune
from ray.train import Checkpoint,RunConfig
from ray.tune.search.hyperopt import HyperOptSearch

class SubNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates, normorder='first', activation='relu'):
        super(SubNetwork, self).__init__()

        # Figure out the shaping of the hidden network
        self.size_array = [input_size]
        self.size_array.extend(hidden_sizes)

        # Make the list of linear transformers based on number of layers
        self.fc = [nn.Linear(self.size_array[ii], self.size_array[ii+1]) for ii in range(len(hidden_sizes))]
        
        # Make the list of dropouts based on number of layers
        self.dropout = [nn.Dropout(p=dropout_rates[ii]) for ii in range(len(hidden_sizes))]

        # Define the normalization layers
        self.bn = [nn.BatchNorm1d(isize) for isize in hidden_sizes]

        # Define the possible activation layers
        self.relu    = nn.ReLU()
        self.tanh    = nn.Tanh()

        # Handle selection of activation layer
        self.normorder = normorder
        if activation == 'relu':
            self.activation_layer = self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh
    
    def forward(self, x):

        # Apply the forward transforms
        for idx,ifc in enumerate(self.fc):
            x = ifc(x)

            if self.normorder == 'first':
                x = self.bn[idx](x)
                x = self.activation_layer(x)
            else:
                x = self.activation_layer(x)
                x = self.bn[idx](x)   

            x = self.dropout[idx](x)
        return x

class CombinedNetwork(nn.Module):
    def __init__(self, input_dict, hidden_dict, dropout_dict, combination_dict, output_size, normorder='first',activation='relu'):
        super(CombinedNetwork, self).__init__()

        # Loop over the training blocks to make the subnetworks
        self.subnets        = []
        input_combined_size = 0
        for iblock in input_dict.keys():
            self.subnets.append(SubNetwork(input_dict[iblock], hidden_dict[iblock], dropout_dict[iblock]))
            input_combined_size += hidden_dict[iblock][-1]

        # Make the list of linear transformers based on number of layers
        hidden_sizes    = combination_dict['hidden']
        dropouts        = combination_dict['dropout']
        self.size_array = [input_combined_size]
        self.size_array.extend(hidden_sizes)
        self.fc = [nn.Linear(self.size_array[ii], self.size_array[ii+1]) for ii in range(len(hidden_sizes))]

        # Make the list of dropouts based on number of layers
        self.dropout = [nn.Dropout(p=dropouts[ii]) for ii in range(len(hidden_sizes))]

        # Define the normalization layers
        self.bn = [nn.BatchNorm1d(isize) for isize in hidden_sizes]

        # Remaining combination layersFinal layers after combining the sub-networks
        output_hsize     = hidden_sizes[-1]
        self.relu        = nn.ReLU()
        self.tanh        = nn.Tanh()
        self.fc_output   = nn.Linear(output_hsize, output_size)
        self.softmax     = nn.Softmax(dim=1)
        self.sigmoid     = nn.Sigmoid()

        # Handle variable normalization order and activation layer
        self.normorder = normorder
        if activation == 'relu':
            self.activation_layer = self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh

    def forward(self, *input_vectors):

        # Get the subnet results
        subnets_out = []
        for ii,itensor in enumerate(input_vectors[:-1]):
            subnets_out.append(self.subnets[ii](itensor))

        # Concatenate the outputs of the three sub-networks
        combined = torch.cat(subnets_out, dim=1)

        # Apply the forward transforms
        for idx,ifc in enumerate(self.fc):

            # Pass the combined output through the final layers
            combined = ifc(combined)

            if self.normorder == 'first':
                combined = self.bn[idx](combined)
                combined = self.activation_layer(combined)
            else:
                combined = self.activation_layer(combined)        
                combined = self.bn[idx](combined)
            
            output   = self.fc_output(combined)
            output   = self.dropout[idx](output)
            output   = self.sigmoid(output)
            print(output)


class ConsensusNetwork(nn.Module):
    def __init__(self, config, input_dict, hidden_dict, dropout_dict, combination_dict, output_size):
        super(ConsensusNetwork, self).__init__()

        self.config           = config
        self.input_dict       = input_dict
        self.hidden_dict      = hidden_dict
        self.dropout_dict     = dropout_dict
        self.combination_dict = combination_dict
        self.output_size      = output_size

    def forward(self):
        CombinedNetwork(self.input_dict, self.hidden_dict, self.dropout_dict, self.combination_dict, self.output_size, normorder=self.config['normorder'], activation=self.config['activation'])
        exit()

def train_pnes(config,DL_object):

    # Unpack the data for our model
    model_block       = DL_object[0]
    train_transformed = DL_object[1]
    test_transformed  = DL_object[2]

    # Make the tensor objects
    train_datasets      = {}
    test_datasets       = {}
    input_dict          = {}
    hidden_dict         = {}
    dropout_dict        = {}
    subnetwork_size_out = 0
    for iblock in model_block:

        # Get the correct column subset
        cols = model_block[iblock]

        # Exogenous variable handling for subnetworks
        if iblock != 'passthrough' and iblock != 'target':

            # Add the data to the train and test data loaders
            train_datasets[iblock] = torch.from_numpy(train_transformed[cols].values.astype(np.float32))
            test_datasets[iblock]  = torch.from_numpy(test_transformed[cols].values.astype(np.float32))

            # Make the subnetwork configuration
            input_dict[iblock] = len(cols)
            nlayer             = config[iblock]['nlayer']
            if nlayer > 0:
                hsizes = []
                dsizes = []
                for ilayer in range(nlayer):
                    hsizes.append(int(config[iblock][f"hsize_{ilayer+1}"]*len(cols)))
                    dsizes.append(int(config[iblock][f"drop_{ilayer+1}"]))
            hidden_dict[iblock]  = hsizes
            dropout_dict[iblock] = dsizes
            subnetwork_size_out += hsizes[-1]
        elif iblock == 'target':
            train_targets = torch.from_numpy(train_transformed[cols].values.astype(np.float32))
            test_targets  = torch.from_numpy(test_transformed[cols].values.astype(np.float32))
            output_size   = len(cols)

    # Define the combined network
    nlayer = config['combined']['nlayer']
    hsizes = []
    dsizes = []
    for ilayer in range(nlayer):
        hsizes.append(int(config['combined'][f"hsize_{ilayer+1}"]*subnetwork_size_out))
        dsizes.append(int(config['combined'][f"drop_{ilayer+1}"]))
    combination_dict = {'hidden':hsizes,'dropout':dsizes}

    # Make the datasets
    train_tensor_dataset = TensorDataset(*train_datasets.values(),train_targets)
    test_tensor_dataset  = TensorDataset(*test_datasets.values(),test_targets)

    # Manage the dataloading
    train_loader  = DataLoader(train_tensor_dataset, batch_size=config['batchsize'], shuffle=True)

    # Make the model
    #model = CombinedNetwork(input_dict, hidden_dict, dropout_dict, combination_dict, output_size, normorder=config['normorder'], activation=config['activation'])
    model = ConsensusNetwork(config, input_dict, hidden_dict, dropout_dict, combination_dict, output_size)

    # Define the loss criterion
    criterion = nn.BCELoss()

    # Select the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        for ibatch in train_loader:
            optimizer.zero_grad()
            outputs = model(*ibatch)
            #loss    = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()

            print(outputs)
            exit()

class tuning_manager:

    def __init__(self,DL_object,ncpu):
        """
        Initialize the tuning manager class. It creates the initial parameter space and kicks off the subprocesses.
        """

        # Save variables from the front end
        self.DL_object         = DL_object
        self.model_block       = DL_object[0]
        self.train_transformed = DL_object[1]
        self.test_transformed  = DL_object[2]
        self.ncpu              = ncpu
        self.ntrial            = 2
        self.raydir            = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/FEATURE_EXTRACTION/DEV/HYPERPARAMTER_TUNING/RAY/'

    def make_tuning_config_mlp(self):
        """
        Define how the parameter space is explored using Ray Tuning.
        """

        # Make the config object
        self.config = {}

        # Create a list of sub-networks we need to define networks for
        self.subnetwork_list = list(self.model_block.keys())
        self.subnetwork_list.append('combination')

        # Define the block specific options
        for iblock in self.subnetwork_list:

            # Create a configuration block for the current set of inputs
            self.config[iblock] = {}

            # Hidden size selection methods. Currently limiting the max number of layers to three
            self.config[iblock]['nlayer']  = tune.randint(0, 3)
            self.config[iblock]["hsize_1"] = tune.uniform(0.05, 1.5)
            self.config[iblock]["hsize_2"] = tune.uniform(0.05, 1.5)
            self.config[iblock]["hsize_3"] = tune.uniform(0.3, 1.5)

            # Drouput fraction selection methods. Currently limiting the max number of layers to three (pairs to the hidden size networks)
            self.config[iblock]["drop_1"] = tune.quniform(0.05, .5, .05)
            self.config[iblock]["drop_2"] = tune.quniform(0.05, .5, .05)
            self.config[iblock]["drop_3"] = tune.quniform(0.05, .5, .05)

        # Global fitting criteria selection
        self.config['lr']         = tune.loguniform(1e-5,1e-3)
        self.config['batchsize']  = tune.choice([32,64,128,256])
        self.config['normorder']  = tune.choice(['before','after'])
        self.config['activation'] = tune.choice(['relu','tanh'])

    def run_ray_tune_mlp(self,nlayer_guess=1,h1guess=1.0,h2guess=1.0,h3guess=1.0,drop1guess=0.4,drop2guess=0.4,drop3guess=0.2,batchguess=64,lrguess=5e-5):
        
        # Define the starting parameters for the global parameters
        current_best_params = [{'lr':lrguess,
                                'batchsize':batchguess,
                                'normorder':'before',
                                'activation': "relu"}]

        # Add in the model block specific parameters
        for iblock in self.subnetwork_list:
            current_best_params[0][iblock] = {'nlayer':nlayer_guess, 'hsize_1':h1guess, 'hsize_2':h2guess,
                                              'hsize_3':h3guess, 'drop_1':drop1guess, 'drop_2':drop2guess,
                                              'drop_3':drop3guess}

        # Define the search parameters
        hyperopt_search = HyperOptSearch(metric="Train_AUC", mode="max", points_to_evaluate=current_best_params)

        # Set the number of cpus to use
        trainable_with_resources = tune.with_resources(train_pnes, {"cpu": self.ncpu})
        trainable_with_parameters = tune.with_parameters(trainable_with_resources, data=(self.DL_object))

        # Create the tranable object
        tuner = tune.Tuner(trainable_with_parameters,param_space=self.config,
                           tune_config=tune.TuneConfig(num_samples=self.ntrial, search_alg=hyperopt_search),
                           run_config=RunConfig(storage_path=self.raydir, name="pnes_experiment"))

        # Get the hyper parameter search results
        results   = tuner.fit()