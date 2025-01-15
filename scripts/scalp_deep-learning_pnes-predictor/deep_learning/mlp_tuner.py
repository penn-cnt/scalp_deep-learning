import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Torch loaders
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Ray imports
import tempfile
from ray import train, tune
from ray.train import Checkpoint,RunConfig
from ray.tune.search.hyperopt import HyperOptSearch

class SubNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates, normorder='first', activation='relu'):
        super(SubNetwork, self).__init__()

        if len(hidden_sizes) > 0:
            # Make a flag to say we need to use subnetwork
            self.passflag = False

            # Figure out the shaping of the hidden network
            self.size_array = [input_size]
            self.size_array.extend(hidden_sizes)

            # Make the list of linear transformers based on number of layers
            self.fc = nn.ModuleList([nn.Linear(self.size_array[ii], self.size_array[ii+1]) for ii in range(len(hidden_sizes))])
            
            # Make the list of dropouts based on number of layers
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout_rates[ii]) for ii in range(len(hidden_sizes))])

            # Define the normalization layers
            self.bn = nn.ModuleList([nn.BatchNorm1d(isize) for isize in hidden_sizes])

            # Define the possible activation layers
            self.relu    = nn.ReLU()
            self.tanh    = nn.Tanh()

            # Handle selection of activation layer
            self.normorder = normorder
            if activation == 'relu':
                self.activation_layer = self.relu
            elif activation == 'tanh':
                self.activation_layer = self.tanh
        else:
            self.passflag = True
    
    def forward(self, x):

        if not self.passflag:
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
        else:
            return x

class CombinedNetwork(nn.Module):
    def __init__(self, input_dict, hidden_dict, dropout_dict, combination_dict, output_size, normorder='first',activation='relu'):
        super(CombinedNetwork, self).__init__()

        # Loop over the training blocks to make the subnetwork architecture
        input_combined_size = 0
        self.subnets        = nn.ModuleList([])
        for iblock in input_dict.keys():

            # Store sizing of subnetwork outputs
            if len(hidden_dict[iblock]) > 0:
                input_combined_size += hidden_dict[iblock][-1]
            else:
                input_combined_size += input_dict[iblock]

            # Make the subnetwork object
            self.subnets.append(SubNetwork(input_dict[iblock], hidden_dict[iblock], dropout_dict[iblock]))

        # Make the list of linear transformers based on number of layers
        hidden_sizes    = combination_dict['hidden']
        dropouts        = combination_dict['dropout']
        self.size_array = [input_combined_size]
        self.size_array.extend(hidden_sizes)
        self.fc = nn.ModuleList([nn.Linear(self.size_array[ii], self.size_array[ii+1]) for ii in range(len(hidden_sizes))])

        # Make the list of dropouts based on number of layers
        self.dropout = nn.ModuleList([nn.Dropout(p=dropouts[ii]) for ii in range(len(hidden_sizes))])

        # Define the normalization layers
        self.bn = nn.ModuleList([nn.BatchNorm1d(isize) for isize in hidden_sizes])

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
        subnet_tensors = []
        for idx,itensor in enumerate(input_vectors[0]):
            idata = self.subnets[idx](itensor)
            subnet_tensors.append(idata)

        # Concatenate the outputs of the three sub-networks
        combined = torch.cat(subnet_tensors, dim=1)

        # Apply the forward transforms
        for idx,ifc in enumerate(self.fc):

            # Pass the combined output through the final layers
            combined = ifc(combined)

            # Some possible hidden network architecture
            if self.normorder == 'first':
                combined = self.bn[idx](combined)
                combined = self.activation_layer(combined)
            else:
                combined = self.activation_layer(combined)        
                combined = self.bn[idx](combined)
            
            # Add a dropput
            combined = self.dropout[idx](combined)
        
        # Final logits to prob transform
        output = self.fc_output(combined)
        output = self.sigmoid(output)

        return output

class ConsensusNetwork(nn.Module):
    def __init__(self, config, input_dict, hidden_dict, dropout_dict, combination_dict, output_size, n_batches, clip_level):
        super(ConsensusNetwork, self).__init__()

        self.config           = config
        self.input_dict       = input_dict
        self.hidden_dict      = hidden_dict
        self.dropout_dict     = dropout_dict
        self.combination_dict = combination_dict
        self.output_size      = output_size
        self.n_batches        = n_batches
        self.combine_net      = CombinedNetwork(self.input_dict, self.hidden_dict, self.dropout_dict, self.combination_dict, self.output_size, normorder=self.config['normorder'], activation=self.config['activation'])
        self.clip_level       = clip_level

        self.fc      = nn.Linear(9, output_size)
        self.relu    = nn.ReLU()
        self.bn      = nn.BatchNorm1d(output_size)
        self.dropout = nn.Dropout(p=0.1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, epoch, categoricals, uids, targets):

        # Send the data to the subnetworks
        outputs = []
        for ibatch in x:
            outputs.append(self.combine_net(*ibatch))

        # Make a new input tensor with all of the clips
        clip_predictions = torch.cat(outputs,dim=0)

        # Return the clip level predictions if requested. Good for testing
        if self.clip_level:
            return clip_predictions, targets

        # Get the patient level features
        patient_train_features,patient_train_labels = reshape_clips_to_patient_level(clip_predictions,categoricals,uids,labels=targets)
        patient_train_features                      = patient_train_features.float()
        
        # Advance the consensus network
        x = self.fc(patient_train_features)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        return x, patient_train_labels

def reshape_clips_to_patient_level(clip_predictions,categorcial_block,uid_indices,labels=None):

    # Make a pandas dataframe of sleep stage, prediction, and uid so we can use pandas groupby for easier restructure
    patient_features  = []
    patient_labels    = []
    for unique_uid in uid_indices.keys():
        
        # Break up the user ids and get the indices we need to subslice
        unique_uid_indices = uid_indices[unique_uid]
        categorical_by_uid = categorcial_block[unique_uid_indices]
        prediction_by_uid  = clip_predictions[unique_uid_indices]

        # Make the user level labels for training datasets
        if labels != None:
            label_by_uid = labels[unique_uid_indices]
            patient_labels.append(label_by_uid[0])

        # Loop over the sleep stages
        predictions_by_categorical = []
        for idx in range(categorical_by_uid.shape[1]):
            
            # Get the rows in this subslice that match the given sleep stage
            cat_inds = (categorical_by_uid[:,idx]==1)

            # If we found data for this sleep stage, calculate posterior distribution
            if cat_inds.sum() > 1:

                # Store the fractional number of samples
                fractional_sample_size = cat_inds.sum()/cat_inds.size(0)

                # Get the list of priors
                prior_predictions = prediction_by_uid[cat_inds]
                
                # Find the most predictive entires
                diffs     = torch.abs(torch.diff(prior_predictions,axis=1))
                diff_inds = (diffs>.1).squeeze()

                # Get the posterior distribution
                prior_predictions_log       = prior_predictions[diff_inds].log()
                prior_predictions_log_joint = prior_predictions_log.sum(dim=0)
                log_norm_factor             = torch.logsumexp(prior_predictions_log_joint, dim=0)
                log_posteriors              = prior_predictions_log_joint - log_norm_factor
                posterior_predictions       = log_posteriors.exp()

                # Append the predictions to the new output object
                predictions_by_categorical.extend([posterior_predictions])
                predictions_by_categorical.append(fractional_sample_size.unsqueeze(0))
            else:
                placeholder = torch.tensor([0,0,0])
                predictions_by_categorical.append(placeholder)
        
        # Store the patient level features to a list
        patient_features.append(torch.cat(predictions_by_categorical))
    
    # Make the patient level features into a tensor object
    patient_features = torch.cat([t.unsqueeze(0) for t in patient_features],dim=0)

    # If working with training data, make the label object a tensor object
    if labels != None:
        patient_labels = torch.cat([t.unsqueeze(0) for t in patient_labels],dim=0)

    return patient_features,patient_labels

def train_pnes(config,DL_object,patient_level=False,raytuning=True,clip_checkpoint_path=None):
    """
    Function that manages the workflow for the MLP model.
    """

    # Unpack the data for our model
    model_block       = DL_object[0]
    train_transformed = DL_object[1]
    test_transformed  = DL_object[2]

    # Define the dictionaries that store the different network options
    train_datasets      = {}
    test_datasets       = {}
    input_dict          = {}
    hidden_dict         = {}
    dropout_dict        = {}
    subnetwork_size_out = 0

    # Make the UID tensor for the final consensus score
    uid_train_indices = train_transformed.groupby(['uid']).indices
    uid_test_indices  = test_transformed.groupby(['uid']).indices

    # Get the output info
    outcols       = model_block['target']
    train_arr     = train_transformed[outcols].values.astype(np.float32)
    test_arr      = test_transformed[outcols].values.astype(np.float32)
    train_targets = torch.from_numpy(train_arr)
    test_targets  = torch.from_numpy(test_arr)
    output_size   = len(outcols)

    # Make the tensor objects for the main blocks
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
            nlayer             = config[f"{iblock}_nlayer"]
            hsizes = []
            dsizes = []
            if nlayer > 0:
                for ilayer in range(nlayer):

                    # Calculate the current size
                    hidden_size = int(config[f"{iblock}_hsize_{ilayer+1}"]*len(cols))

                    # Ensure a minimum size of one output neuron
                    if hidden_size == 0:
                        hidden_size = 1

                    # Calculate the dropout size
                    if hidden_size > output_size:
                        drop_size = float(config[f"{iblock}_drop_{ilayer+1}"])
                    else:
                        drop_size = 0

                    # Store the sizes
                    hsizes.append(hidden_size)
                    dsizes.append(drop_size)
                    final_layer_size = hsizes[-1]
            else:
                final_layer_size = len(cols)

            # Store the info about the hidden layer network
            hidden_dict[iblock]  = hsizes
            dropout_dict[iblock] = dsizes
            subnetwork_size_out += final_layer_size

    # Define the combined network
    nlayer = config['combined_nlayer']
    hsizes = []
    dsizes = []
    for ilayer in range(nlayer):

        # Calculate the current hidden size
        hidden_size = int(config[f"combined_hsize_{ilayer+1}"]*subnetwork_size_out)

        # Ensure a minimum size of the output neurons
        if hidden_size < output_size:
            hidden_size = output_size 

        # Calculate the dropout size
        if hidden_size > output_size:
            drop_size = float(config[f"combined_drop_{ilayer+1}"])
        else:
            drop_size = 0

        # Store the info about the hidden layer network
        hsizes.append(hidden_size)
        dsizes.append(drop_size)
    combination_dict = {'hidden':hsizes,'dropout':dsizes}

    # Make the datasets
    train_tensor_dataset = TensorDataset(*train_datasets.values(),train_targets)
    test_tensor_dataset  = TensorDataset(*test_datasets.values(),test_targets)

    # Manage the dataloading
    train_loader  = DataLoader(train_tensor_dataset, batch_size=config['batchsize'], shuffle=True)

    # Make the model
    combine_model = CombinedNetwork(input_dict,hidden_dict,dropout_dict,combination_dict,output_size,normorder=config['normorder'], activation=config['activation'])

    # Define the loss criterion
    sums              = train_targets.numpy().sum(axis=0)
    pos_weight        = config['weight']*torch.tensor([sums[0]/sums[1]])
    patient_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Select the optimizer
    combine_optimizer   = optim.Adam(combine_model.parameters(), lr=config['lr'])

    # If we were passed a checkpoint, load it now
    if clip_checkpoint_path != None:
        checkpoint = torch.load(clip_checkpoint_path)
        combine_model.load_state_dict(checkpoint['model_state_dict'])
        combine_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # Train the combination model
        num_epochs  = 10
        for epoch in tqdm(range(num_epochs), total=num_epochs, disable=np.invert(raytuning)):
            combine_model.train()
            for ibatch in train_loader:
                
                # Kick off the consensus handler
                combine_optimizer.zero_grad()

                # Unpack the batch
                labels       = ibatch[-1]
                batchtensors = ibatch[:-1]

                # get the output for the current batch
                outputs = combine_model(batchtensors)
                loss    = patient_criterion(outputs, labels)
                loss.backward()
                combine_optimizer.step()

    # Evaluate the combination model
    combine_model.eval()

    # Get the clip level predictions
    outputs = combine_model([*train_datasets.values()])

    # Get the predicted outputs
    y_pred       = outputs.squeeze().detach().numpy()
    y_pred_max   = (y_pred[:,1]>y_pred[:,0]).astype(int).reshape((-1,1))
    y_pred_clean = np.hstack((1-y_pred_max, y_pred_max))
    y_pred_max   = np.argmax(y_pred_clean,axis=1)

    # Get the measured outputs in useable format
    y_meas_clean = train_arr
    y_meas_max   = np.argmax(y_meas_clean,axis=1)

    # Measure the accuracy
    train_acc  = (y_pred_max==y_meas_max).sum()/y_meas_max.size

    # Measure the auc
    train_auc = roc_auc_score(y_meas_clean,y_pred_clean)

    # Make a checkpoint for RAY tuning
    if raytuning:
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = {'model': combine_model.state_dict(),'optimizer': combine_optimizer.state_dict()}
            torch.save(checkpoint,os.path.join(temp_checkpoint_dir, "model.pth"))
            train.report({"Train_AUC": train_auc, "Train_ACC": train_acc}, checkpoint=checkpoint)
    else:
        print("Acc:",train_acc)
        print("AUC:",train_auc)
    
class tuning_manager:

    def __init__(self, DL_object, ncpu, ntrial, outfile, raydir, hotconfig):
        """
        Initialize the tuning manager class. It creates the initial parameter space and kicks off the subprocesses.
        """

        # Set the random seed
        torch.manual_seed(42)

        # Save variables from the front end
        self.DL_object         = DL_object
        self.model_block       = DL_object[0]
        self.train_transformed = DL_object[1]
        self.test_transformed  = DL_object[2]
        self.ncpu              = ncpu
        self.ntrial            = ntrial
        self.raydir            = raydir
        self.outfile           = outfile
        self.hotconfig         = hotconfig

    def make_tuning_config_mlp(self):
        """
        Define how the parameter space is explored using Ray Tuning.
        """

        # Make the config object
        self.config = {}

        # Create a list of sub-networks we need to define networks for
        self.subnetwork_list = []
        for iblock in self.model_block.keys():
            if iblock not in ['target','passthrough']:
                self.subnetwork_list.append(iblock)

        # Define the block specific options
        for iblock in self.subnetwork_list:

            # Hidden size selection methods. Currently limiting the max number of layers to three
            self.config[f"{iblock}_nlayer"]  = tune.randint(0, 4)
            self.config[f"{iblock}_hsize_1"] = tune.quniform(0.05, 1.5, .05)
            self.config[f"{iblock}_hsize_2"] = tune.quniform(0.05, 1.5, .05)
            self.config[f"{iblock}_hsize_3"] = tune.quniform(0.05, 1.5, .05)

            # Drouput fraction selection methods. Currently limiting the max number of layers to three (pairs to the hidden size networks)
            self.config[f"{iblock}_drop_1"] = tune.quniform(0.05, .5, .05)
            self.config[f"{iblock}_drop_2"] = tune.quniform(0.05, .5, .05)
            self.config[f"{iblock}_drop_3"] = tune.quniform(0.05, .5, .05)

        # Define the combination configuration block
        self.config[f"combined_nlayer"]  = tune.randint(1, 4)
        self.config[f"combined_hsize_1"] = tune.quniform(0.05, 1.5, .05)
        self.config[f"combined_hsize_2"] = tune.quniform(0.05, 1.5, .05)
        self.config[f"combined_hsize_3"] = tune.quniform(0.05, 1.5, .05)
        self.config[f"combined_drop_1"]  = tune.quniform(0.05, .5, .05)
        self.config[f"combined_drop_2"]  = tune.quniform(0.05, .5, .05)
        self.config[f"combined_drop_3"]  = tune.quniform(0.05, .5, .05)

        # Global fitting criteria selection
        self.config['lr']         = tune.loguniform(1e-5,1e-3)
        self.config['batchsize']  = tune.choice([64,128,256,512])
        self.config['normorder']  = tune.choice(['before','after'])
        self.config['activation'] = tune.choice(['relu','tanh'])
        self.config['weight']     = tune.loguniform(1,1e5)

    def run_ray_tune_mlp(self,coldstart=False,nlayer_guess=1,h1guess=1.0,h2guess=1.0,h3guess=1.0,drop1guess=0.4,drop2guess=0.4,drop3guess=0.2,batchguess=64,lrguess=5e-5):
        
        # Define the starting parameters for the global parameters
        if coldstart:
            current_best_params = [{'lr':lrguess,
                                    'batchsize':batchguess,
                                    'normorder':'before',
                                    'activation': "relu",
                                    'weight': 1000}]

            # Add in the model block specific parameters
            for iblock in self.subnetwork_list:
                current_best_params[0][f"{iblock}_nlayer"]  = nlayer_guess
                current_best_params[0][f"{iblock}_hsize_1"] = h1guess
                current_best_params[0][f"{iblock}_hsize_2"] = h2guess
                current_best_params[0][f"{iblock}_hsize_3"] = h3guess
                current_best_params[0][f"{iblock}_drop_1"]  = drop1guess
                current_best_params[0][f"{iblock}_drop_2"]  = drop2guess
                current_best_params[0][f"{iblock}_drop_3"]  = drop3guess
            
            # Make the combinatorial guess network
            current_best_params[0][f"combined_nlayer"]  = 1
            current_best_params[0][f"combined_hsize_1"] = 0.8
            current_best_params[0][f"combined_hsize_2"] = 0.8
            current_best_params[0][f"combined_hsize_3"] = 0.8
            current_best_params[0][f"combined_drop_1"]  = 0.1
            current_best_params[0][f"combined_drop_2"]  = 0.1
            current_best_params[0][f"combined_drop_3"]  = 0.1
        else:
            current_best_params = [self.hotconfig]
            
        # Define the search parameters
        hyperopt_search = HyperOptSearch(metric="Train_AUC", mode="max", points_to_evaluate=current_best_params, random_state_seed=42)

        # Set the number of cpus to use
        trainable_with_resources = tune.with_resources(train_pnes, {"cpu": self.ncpu})
        trainable_with_parameters = tune.with_parameters(trainable_with_resources, DL_object=(self.DL_object))

        # Create the tranable object
        tuner = tune.Tuner(trainable_with_parameters,param_space=self.config,
                           tune_config=tune.TuneConfig(num_samples=self.ntrial, search_alg=hyperopt_search),
                           run_config=RunConfig(storage_path=self.raydir, name="pnes_experiment",verbose=1,
                                                failure_config=train.FailureConfig(fail_fast=True)))

        # Get the hyper parameter search results
        results = tuner.fit()

        # Save the tuned results
        result_DF = results.get_dataframe()
        result_DF.to_csv(self.outfile)