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
import ray
import tempfile
from ray import train, tune
from ray.train import Checkpoint,RunConfig
from ray.tune.search.hyperopt import HyperOptSearch

class SubNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rates, activation, normorder, normtype):
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
            if activation == 'relu':
                self.activation_layer = nn.ReLU()
            elif activation == 'tanh':
                self.activation_layer = nn.Tanh()
            elif activation == 'sigmoid':
                self.activation_layer = nn.Sigmoid()
            elif activation == 'softmax':
                self.activation_layer = nn.Softmax()
            elif activation == 'lrelu':
                self.activation_layer = nn.LeakyReLU()
            elif activation == 'pass':
                self.activation_layer = nn.Identity()

            # Define the possible normalization layers
            self.normorder = normorder
            if normtype == 'batch1d':
                self.bn = nn.ModuleList([nn.BatchNorm1d(isize) for isize in hidden_sizes])
            elif normtype == 'pass':
                self.bn = nn.ModuleList([nn.Identity(isize) for isize in hidden_sizes])

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
    def __init__(self, input_dict, hidden_dict, dropout_dict, activation_dict, norm_dict, combination_dict, output_size):
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
            self.subnets.append(SubNetwork(input_dict[iblock], hidden_dict[iblock], dropout_dict[iblock], activation_dict[iblock], norm_dict[iblock]['order'], norm_dict[iblock]['type']))

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
        self.fc_output   = nn.Linear(output_hsize, output_size)

        # Handle variable normalization order and activation layer
        activation = combination_dict['activation']
        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'tanh':
            self.activation_layer = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation_layer = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation_layer = nn.Softmax()
        elif activation == 'lrelu':
            self.activation_layer = nn.LeakyReLU()
        elif activation == 'pass':
            self.activation_layer = nn.Identity()
        self.out_activation = nn.Sigmoid()

        # Define the possible normalization layers
        self.normorder = combination_dict['norm_order']
        normtype       = combination_dict['normalization']
        if normtype == 'batch1d':
            self.bn = nn.ModuleList([nn.BatchNorm1d(isize) for isize in hidden_sizes])
        elif normtype == 'pass':
            self.bn = nn.ModuleList([nn.Identity(isize) for isize in hidden_sizes])

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
        output = self.out_activation(output)

        return output

class ConsensusNetwork(nn.Module):
    def __init__(self, consensus_dict, input_size, output_size):
        super(ConsensusNetwork, self).__init__()

        # Extract from input info
        hidden_sizes     = consensus_dict['hidden']
        dropout_rates    = consensus_dict['dropout']

        # Figure out the shaping of the hidden network
        self.size_array = [input_size]
        self.size_array.extend(hidden_sizes)

        # Make the list of linear transformers based on number of layers
        self.fc        = nn.ModuleList([nn.Linear(self.size_array[ii], self.size_array[ii+1]) for ii in range(len(hidden_sizes))])
        self.fc_output = nn.Linear(self.size_array[-1], output_size)
        
        # Make the list of dropouts based on number of layers
        self.dropout = nn.ModuleList([nn.Dropout(p=dropout_rates[ii]) for ii in range(len(hidden_sizes))])

        # Define the possible activation layers
        activation = consensus_dict['activation']
        if activation == 'relu':
            self.activation_layer = nn.ReLU()
        elif activation == 'tanh':
            self.activation_layer = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation_layer = nn.Sigmoid()
        elif activation == 'softmax':
            self.activation_layer = nn.Softmax()
        elif activation == 'lrelu':
            self.activation_layer = nn.LeakyReLU()
        elif activation == 'pass':
            self.activation_layer = nn.Identity()
        self.out_activation = nn.Sigmoid()

        # Define the possible normalization layers
        self.normorder = consensus_dict['norm_order']
        normtype       = consensus_dict['normalization']
        if normtype == 'batch1d':
            self.bn = nn.ModuleList([nn.BatchNorm1d(isize) for isize in hidden_sizes])
        elif normtype == 'pass':
            self.bn = nn.ModuleList([nn.Identity(isize) for isize in hidden_sizes])

    def forward(self, x):

        for idx,ifc in enumerate(self.fc):
            x = ifc(x)

            if self.normorder == 'first':
                x = self.bn[idx](x)
                x = self.activation_layer(x)
            else:
                x = self.activation_layer(x)
                x = self.bn[idx](x)   

            x = self.dropout[idx](x)

        # Final output layer
        x = self.fc_output(x)
        x = self.out_activation(x)

        return x

class clip_to_consensus:

    def __init__(self):
        pass

    def handler(self):

        self.weight_method    = 'sleep_stage'
        self.prob_method      = 'quantile'
        self.thresholds       = [0.8,0.8,0.8]
        self.reference_tensor = None

        #print(self.model_block['categorical'])
        #exit()

        # Get indices for each patient, structured to allow for weighting or passing all indices to some probability vector method
        if self.weight_method == None:
            train_inds, test_inds = self.weighting_none()
        elif self.weight_method == 'sleep_stage':
            train_inds, test_inds = self.weighting_sleep_stage()

        # Get the new consensus tensors
        train_features,train_targets = self.make_consensus_tensor(train_inds,self.clip_training_predictions_tensor,self.train_targets)
        test_features,test_targets   = self.make_consensus_tensor(test_inds,self.clip_testing_predictions_tensor,self.test_targets)

        return train_features,train_targets,test_features,test_targets

    ########################################################
    ### Creation of a new tensor with correct leaf nodes ###
    ########################################################
    
    def make_consensus_tensor(self,input_inds,input_predictions,input_targets):

        # Get the probability vector for training data loop over user and possible weighting structure
        consensus_posterior_raw = {ikey:[] for ikey in input_inds.keys()}
        consensus_weighting_raw = {ikey:[] for ikey in input_inds.keys()}
        consensus_target_raw    = {ikey:[] for ikey in input_inds.keys()}
        for ikey in input_inds.keys():

            # Get the current user indices
            consensus_ind_slice = input_inds[ikey]

            # Add the targets to the tracking dictionary
            tind                       = [jval for ival in consensus_ind_slice.values() for jval in ival][0]
            consensus_target_raw[ikey] = input_targets[tind]

            # Loop over possible weighting axis
            for jkey in consensus_ind_slice.keys():

                # Get the current user ids for this weight
                current_user_inds = consensus_ind_slice[jkey]
                
                # Get the fraction of all patient entires with this weighting type
                fraction                      = torch.tensor([len(consensus_ind_slice[jkey])/sum([len(ival) for ival in consensus_ind_slice.values()])])
                consensus_weighting_raw[ikey].append(fraction)

                # Get the list of priors
                prior_predictions = input_predictions[current_user_inds]

                # Get the result using the requested method
                if prior_predictions.shape[0] > 0:
                    if self.prob_method == 'quantile':
                        posterior_prediction = self.quantile(prior_predictions,threshold=self.thresholds[jkey])
                        if self.reference_tensor == None:self.reference_tensor=torch.zeros_like(posterior_prediction)
                else:
                    posterior_prediction = None
                
                # Add this info to the tracking dictionary
                consensus_posterior_raw[ikey].append(posterior_prediction)    
        
        # Update any missing entries with the back gradient compatible zero tensor. Add weights
        for ikey,ivalue in consensus_posterior_raw.items():
            for jdx,jvalue in enumerate(ivalue):
                if jvalue == None:
                    consensus_posterior_raw[ikey][jdx] = self.reference_tensor
                consensus_posterior_raw[ikey][jdx] = torch.cat([consensus_posterior_raw[ikey][jdx],consensus_weighting_raw[ikey][jdx]])

        # Convert features to a tensor
        consensus_posterior_raw_list = list(consensus_posterior_raw.values())
        consensus_features           = torch.stack([torch.cat(row, dim=0) for row in consensus_posterior_raw_list],dim=0)

        # Convert targets to a tensor
        consensus_targets_raw_list = [[ival] for ival in consensus_target_raw.values()]
        consensus_targets          = torch.stack([torch.cat(row, dim=0) for row in consensus_targets_raw_list],dim=0)

        return consensus_features,consensus_targets

    #################################################
    ### Methods for creating the consensus vector ###
    #################################################
    def weighting_sleep_stage(self):
        """
        Returns a final vector for each patient with P(PNES),P(Epilepsy) for each sleep stage.
        It also returns a N(Sleep Stage Clips)/N(Patient Total Clips) for weighting the relative importance of each stage for a given patient.
        """

        # Some clean up of attention inputs down to a singular consensus
        if self.attention:
            # Get the current sleep stage if using attention vector
            mid_ind                      = self.n_attention//2
            centered_categorical_indices = []
            for ilabel in self.model_block['categorical']:
                print(ilabel)
                if int(ilabel.split('_')[-1])==mid_ind:
                    centered_categorical_indices.append(True)
                else:
                    centered_categorical_indices.append(False)
            exit()

            # Get the categorical vector centered on the current time point
            train_categorical = self.train_datasets['categorical'][:,centered_categorical_indices]
            test_categorical  = self.test_datasets['categorical'][:,centered_categorical_indices]
        else:
            train_categorical = self.train_datasets['categorical']
            test_categorical  = self.test_datasets['categorical']

        # Create the weighting list for train and test
        train_inds_by_uid = {}
        test_inds_by_uid  = {}
        for ikey in self.uid_train_indices.keys():
            train_inds_by_uid[ikey] = {icol:[] for icol in range(train_categorical.shape[1])}
        for ikey in self.uid_test_indices.keys():
            test_inds_by_uid[ikey] = {icol:[] for icol in range(test_categorical.shape[1])}

        # Populate the weighted train index list
        for ii,uid_key in enumerate(list(self.uid_train_indices.keys())):

            # Get all the indices for this patient
            all_train_uid_indices = self.uid_train_indices[uid_key]

            # Get the categorical slice
            train_categorical_slice = train_categorical[all_train_uid_indices]

            # Loop over each individual indes to figure out which categorical bin to put it in
            for jj,uid_key_jj in enumerate(all_train_uid_indices):

                # Get the categorical index for sorting
                cat_ind = np.where(train_categorical_slice[jj]==1)[0][0]

                # Append the index to the right categorical index
                train_inds_by_uid[uid_key][cat_ind].append(uid_key_jj)

        # Populate the weighted test index list
        for ii,uid_key in enumerate(list(self.uid_test_indices.keys())):

            # Get all the indices for this patient
            all_test_uid_indices = self.uid_test_indices[uid_key]

            # Get the categorical slice
            test_categorical_slice = test_categorical[all_test_uid_indices]

            # Loop over each individual indes to figure out which categorical bin to put it in
            for jj,uid_key_jj in enumerate(all_test_uid_indices):

                # Get the categorical index for sorting
                cat_ind = np.where(test_categorical_slice[jj]==1)[0][0]

                # Append the index to the right categorical index
                test_inds_by_uid[uid_key][cat_ind].append(uid_key_jj)
        
        return train_inds_by_uid,test_inds_by_uid

    def weighting_none(self):
        """
        No weighting of the probabilities
        """

        # Create the weighting list for train and test
        train_inds_by_uid = {}
        test_inds_by_uid  = {}

        # Populate unweighted train index list
        for uid_key in self.uid_train_indices.keys():
            train_inds_by_uid[uid_key] = {0:list(self.uid_train_indices[uid_key])}

        # Populate unweighted test index list
        for uid_key in self.uid_test_indices.keys():
            test_inds_by_uid[uid_key] = {0:list(self.uid_test_indices[uid_key])}
        
        return train_inds_by_uid,test_inds_by_uid

    ################################################################
    ### Methods for creating a single probability from all clips ###
    ################################################################

    def posterior_selection(self):
        # Find the most predictive entires
        diffs     = torch.abs(torch.diff(prior_predictions,axis=1))
        diff_inds = (diffs>threshold).squeeze()

        # Get the posterior distribution
        prior_predictions_log       = prior_predictions[diff_inds].log()
        prior_predictions_log_joint = prior_predictions_log.sum(dim=0)
        log_norm_factor             = torch.logsumexp(prior_predictions_log_joint, dim=0)
        log_posteriors              = prior_predictions_log_joint - log_norm_factor
        posterior_predictions       = log_posteriors.exp()
        return posterior_predictions

    def quantile_vector(self,input_vector,qvec=[0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]):
        return nn.ModuleList([torch.quantile(input_vector,q=threshold, dim=0) for threshold in qvec])

    def quantile(self,input_vector,threshold):
        return torch.quantile(input_vector,q=threshold, dim=0)

class train_pnes(clip_to_consensus):

    def __init__(self,config,DL_object,patient_level=False,raytuning=True,checkpoint_path=None):
        
        # Save important variables from handler input to the class instance
        self.config            = config
        self.model_block       = DL_object[0]
        self.train_transformed = DL_object[1]
        self.test_transformed  = DL_object[2]
        self.patient_level     = patient_level
        self.raytuning         = raytuning
        self.checkpoint_path   = checkpoint_path

        # Initialize some classwide variables
        self.nepoch_clip         = 20
        self.nepoch_patient      = 20
        self.train_datasets      = {}
        self.test_datasets       = {}
        self.input_dict          = {}
        self.hidden_dict         = {}
        self.dropout_dict        = {}
        self.activation_dict     = {}
        self.norm_dict           = {}
        self.subnetwork_size_out = 0
        self.comb_loss           = []
        self.con_loss            = []
        self.attention           = True
        self.n_attention         = 9

        # Get the user id indices
        self.get_uids()

    def run_basemodel_pipeline(self):
        """
        Manage the workflow for the PNES predictions.
        """

        # Basic setup steps
        self.get_output_size()
        self.prepare_datasets()
        self.prepare_hiddenstates()
        self.make_combine_model()
        self.make_combine_criterion()
        self.make_combine_optimizer()

        # Combination (i.e. Clip level) model
        self.run_combination_model(self.checkpoint_path)

        # Update tensors to return to user for possible downstream analysis
        self.update_tensors_w_probs()

        # Make a consensus tensor
        if self.patient_level:

            # Get the consensus formatted tensors
            packed_results                         = clip_to_consensus.handler(self)
            self.training_consensus_tensor         = packed_results[0]
            self.training_consensus_tensor_targets = packed_results[1]
            self.testing_consensus_tensor          = packed_results[2]
            self.testing_consensus_tensor_targets  = packed_results[3]

            #self.training_consensus_tensor,self.training_consensus_tensor_targets = self.clip_to_patient_transform(self.clip_training_predictions_tensor,self.train_datasets['categorical'],self.uid_train_indices,targets=self.train_targets)
            #self.testing_consensus_tensor,self.testing_consensus_tensor_targets   = self.clip_to_patient_transform(self.clip_testing_predictions_tensor,self.test_datasets['categorical'],self.uid_test_indices,targets=self.test_targets)

            # Make the consensus hidden states
            self.create_consensus_datasets()
            self.prepare_consensus_network()
            self.make_consensus_model()
            self.make_consensus_criterion()
            self.make_consensus_optimizer()

            # Consensus Model
            self.run_consensus_model(self.checkpoint_path)

    def update_basemodel_pipeline(self):
        
        # Basic setup steps
        self.get_output_size()
        self.prepare_datasets()
        self.prepare_hiddenstates()
        self.make_combine_model()
        self.make_combine_criterion()
        self.make_combine_optimizer()

        # Get the first pass predictions for the test dataset
        self.run_combination_model(self.checkpoint_path)

        # Make a copy of the testing dataset to update with new predictions
        self.train_transformed_w_probs = self.train_transformed.copy()
        self.test_transformed_w_probs  = self.test_transformed.copy()
        self.train_transformed_w_probs[['epilepsy','pnes']] = self.clip_training_predictions_tensor.detach().numpy()
        self.test_transformed_w_probs[['epilepsy','pnes']]  = self.clip_testing_predictions_tensor.detach().numpy()

        import pickle
        outpath = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/MODELS/SSL/DATA/traintest_transformed.pickle'
        pickle.dump((self.train_transformed_w_probs,self.test_transformed_w_probs),open(outpath,"wb"))
        exit()
                    
    def get_uids(self):

        # Make the UID tensor for the final consensus score
        self.uid_train_indices = self.train_transformed.groupby(['uid']).indices
        self.uid_test_indices  = self.test_transformed.groupby(['uid']).indices

    def get_output_size(self):
        """
        Define the output hidden layer size.
        """

        # Get the output info
        outcols                 = self.model_block['target']
        self.train_target_array = self.train_transformed[outcols].values.astype(np.float32)
        self.test_target_array  = self.test_transformed[outcols].values.astype(np.float32)
        self.train_targets      = torch.from_numpy(self.train_target_array)
        self.test_targets       = torch.from_numpy(self.test_target_array)
        self.output_size        = len(outcols)

    def prepare_datasets(self):

        # Make the tensor objects for the initial sub
        for iblock in self.model_block:

            # Get the correct column subset
            cols = self.model_block[iblock]

            # Exogenous variable handling for subnetworks
            if iblock != 'passthrough' and iblock != 'target':

                # Add the data to the train and test data loaders
                self.train_datasets[iblock] = torch.from_numpy(self.train_transformed[cols].values.astype(np.float32))
                self.test_datasets[iblock]  = torch.from_numpy(self.test_transformed[cols].values.astype(np.float32))

        # Make the datasets
        self.train_tensor_dataset = TensorDataset(*self.train_datasets.values(),self.train_targets)
        self.test_tensor_dataset  = TensorDataset(*self.test_datasets.values(),self.test_targets)

        # Manage the dataloading
        self.train_loader  = DataLoader(self.train_tensor_dataset, batch_size=int(self.config['batchsize']), shuffle=True)

    def prepare_hiddenstates(self):
        """
        Create the variably sized hidden states and make the tensor objects.
        """

        # Make the tensor objects for the initial sub
        for iblock in self.model_block:

            # Get the correct column subset
            cols = self.model_block[iblock]

            # Exogenous variable handling for subnetworks
            if iblock != 'passthrough' and iblock != 'target':

                # Make the subnetwork configuration
                self.input_dict[iblock] = len(cols)
                nlayer                  = self.config[f"{iblock}_nlayer"]
                hsizes                  = []
                dsizes                  = []
                if nlayer > 0:
                    for ilayer in range(nlayer):

                        # Calculate the current size
                        hidden_size = int(self.config[f"{iblock}_hsize_{ilayer+1}"]*len(cols))

                        # Ensure a minimum size of one output neuron
                        if hidden_size == 0:
                            hidden_size = 1

                        # Calculate the dropout size
                        if hidden_size > self.output_size:
                            drop_size = float(self.config[f"{iblock}_drop_{ilayer+1}"])
                        else:
                            drop_size = 0

                        # Store the sizes
                        hsizes.append(hidden_size)
                        dsizes.append(drop_size)
                        final_layer_size = hsizes[-1]
                else:
                    final_layer_size = len(cols)

                # Store the info about the hidden layer network
                self.hidden_dict[iblock]        = hsizes
                self.dropout_dict[iblock]       = dsizes
                self.activation_dict[iblock]    = self.config[f"{iblock}_activation"]
                self.norm_dict[iblock]          = {}
                self.norm_dict[iblock]['type']  = self.config[f"{iblock}_normalization"]
                self.norm_dict[iblock]['order'] = self.config[f"{iblock}_norm_order"]
                self.subnetwork_size_out       += final_layer_size

        # Define the combined network
        nlayer = self.config['combined_nlayer']
        hsizes = []
        dsizes = []
        for ilayer in range(nlayer):

            # Calculate the current hidden size
            hidden_size = int(self.config[f"combined_hsize_{ilayer+1}"]*self.subnetwork_size_out)

            # Ensure a minimum size of the output neurons
            if hidden_size < self.output_size:
                hidden_size = self.output_size 

            # Calculate the dropout size
            if hidden_size > self.output_size:
                drop_size = float(self.config[f"combined_drop_{ilayer+1}"])
            else:
                drop_size = 0

            # Store the info about the hidden layer network
            hsizes.append(hidden_size)
            dsizes.append(drop_size)
        self.combination_dict = {'hidden':hsizes,'dropout':dsizes,'activation':self.config['combined_activation'],
                                 'normalization':self.config['combined_normalization'],'norm_order':self.config['combined_norm_order']}

    def make_combine_model(self):
        """
        Create the model object pointers.
        """

        self.combine_model = CombinedNetwork(self.input_dict,self.hidden_dict,self.dropout_dict,self.activation_dict,self.norm_dict,self.combination_dict,self.output_size)
 
    def make_combine_criterion(self):
        """
        Make the criterion objects.
        """

        sums                   = self.train_targets.numpy().sum(axis=0)
        self.pos_weight        = self.config['weight']*torch.tensor([sums[0]/sums[1]])
        self.combine_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def make_combine_optimizer(self):
        """
        Make the optimizer objects.
        """

        self.combine_optimizer = optim.Adam(self.combine_model.parameters(), lr=self.config['lr'])

    def get_acc_auc(self,intensor,truth_arr):

        # Get the predicted outputs
        y_pred       = intensor.squeeze().detach().numpy()
        y_pred_max   = (y_pred[:,1]>y_pred[:,0]).astype(int).reshape((-1,1))
        y_pred_clean = np.hstack((1-y_pred_max, y_pred_max))
        y_pred_max   = np.argmax(y_pred_clean,axis=1)

        # Get the measured outputs in useable format
        truth_max = np.argmax(truth_arr,axis=1)

        # Measure the accuracy
        acc  = (y_pred_max==truth_max).sum()/truth_max.size

        # Measure the auc
        auc = roc_auc_score(truth_arr,y_pred_clean)

        return acc,auc,y_pred

    def run_combination_model(self,checkpoint_path):
        
        # If we were passed a checkpoint, load it now. Otherwise, train the combination layer
        if checkpoint_path != None:
            checkpoint = torch.load(checkpoint_path)
            self.combine_model.load_state_dict(checkpoint['combine_model'])
            self.combine_optimizer.load_state_dict(checkpoint['combine_optimizer'])
        else:
            # Train the combination model
            for epoch in tqdm(range(self.config["combination_nepoch"]), total=self.config["combination_nepoch"], disable=self.raytuning):
                self.combine_model.train()
                loss_list = []
                for ibatch in self.train_loader:
                    
                    # Kick off the combine handler
                    self.combine_optimizer.zero_grad()

                    # Unpack the batch
                    labels       = ibatch[-1]
                    batchtensors = ibatch[:-1]

                    # get the output for the current batch
                    outputs = self.combine_model(batchtensors)
                    loss    = self.combine_criterion(outputs, labels)
                    loss_list.append(loss.detach().item())
                    loss.backward()
                    self.combine_optimizer.step()
                self.comb_loss.append(loss_list)

        # Evaluate the combination model
        self.combine_model.eval()

        # Get the clip level predictions
        train_outputs = self.combine_model([*self.train_datasets.values()])
        test_outputs  = self.combine_model([*self.test_datasets.values()])

        # Measure the accuracy
        train_acc_clip, train_auc_clip,y_pred = self.get_acc_auc(train_outputs,self.train_target_array)
        test_acc_clip, test_auc_clip,_        = self.get_acc_auc(test_outputs,self.test_target_array)

        # Make a checkpoint for RAY tuning
        if self.raytuning and not self.patient_level:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                outdict    = {'combine_model': self.combine_model.state_dict(),'combine_optimizer': self.combine_optimizer.state_dict(),'comb_loss':self.comb_loss, 'config':self.config}
                torch.save(outdict,os.path.join(temp_checkpoint_dir, "consensus_model.pth"))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                # Send the current training result back to Tune
                train.report({"Train_AUC": train_auc_clip,"Test_ACC":train_acc_clip}, checkpoint=checkpoint)
        elif not self.raytuning and not self.patient_level:
            print(f"Training Accuracy (Clip): {train_acc_clip:0.3f}")
            print(f"Training AUC      (Clip): {train_auc_clip:0.3f}")
            print(f"Testing Accuracy  (Clip): {test_acc_clip:0.3f}")
            print(f"Testing AUC       (Clip): {test_auc_clip:0.3f}")

        # Store the clip layer predictions to the class instance
        self.train_acc_clip = train_acc_clip
        self.train_auc_clip = train_auc_clip
        self.test_acc_clip  = test_acc_clip
        self.test_auc_clip  = test_auc_clip
        self.clip_training_predictions_tensor = train_outputs
        self.clip_testing_predictions_tensor  = test_outputs
        self.clip_training_predictions_array  = y_pred

    def clip_to_patient_transform(self,clip_predictions,categorcial_block,uid_indices,targets=None):

        def posterior_selection(prior_predictions,threshold):
            # Find the most predictive entires
            diffs     = torch.abs(torch.diff(prior_predictions,axis=1))
            diff_inds = (diffs>threshold).squeeze()

            # Get the posterior distribution
            prior_predictions_log       = prior_predictions[diff_inds].log()
            prior_predictions_log_joint = prior_predictions_log.sum(dim=0)
            log_norm_factor             = torch.logsumexp(prior_predictions_log_joint, dim=0)
            log_posteriors              = prior_predictions_log_joint - log_norm_factor
            posterior_predictions       = log_posteriors.exp()
            return posterior_predictions
        def quantile(prior_predictions,theshold):
            return torch.quantile(prior_predictions,q=theshold, dim=0)

        # Make a pandas dataframe of sleep stage, prediction, and uid so we can use pandas groupby for easier restructure
        patient_features  = []
        patient_targets   = []
        for unique_uid in uid_indices.keys():
            
            # Break up the user ids and get the indices we need to subslice
            unique_uid_indices = uid_indices[unique_uid]
            categorical_by_uid = categorcial_block[unique_uid_indices]
            prediction_by_uid  = clip_predictions[unique_uid_indices]

            # Make the user level labels for training datasets
            if targets != None:
                targets_by_uid = targets[unique_uid_indices]
                patient_targets.append(targets_by_uid[0])

            # Loop over the sleep stages
            predictions_by_categorical = []
            for idx in range(categorical_by_uid.shape[1]):
                
                # Get the categorical column name
                catcol = self.model_block['categorical'][idx]

                # Get the rows in this subslice that match the given sleep stage
                cat_inds = (categorical_by_uid[:,idx]==1)

                # If we found data for this sleep stage, calculate posterior distribution
                if cat_inds.sum() > 1:

                    # Store the fractional number of samples
                    fractional_sample_size = cat_inds.sum()/cat_inds.size(0)

                    # Get the list of priors
                    prior_predictions = prediction_by_uid[cat_inds]
                    
                    # get the output probability for this categorical column
                    if self.config['consensus_theshold_method'] == 'posterior':
                        posterior_predictions = posterior_selection(prior_predictions,self.config[f"consensus_theshold_{catcol}"])
                    elif self.config['consensus_theshold_method'] == 'quantile':
                        posterior_predictions = quantile(prior_predictions,self.config[f"consensus_theshold_{catcol}"])

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
        if targets != None:
            patient_targets = torch.cat([t.unsqueeze(0) for t in patient_targets],dim=0)

        return patient_features,patient_targets
    
    def prepare_consensus_network(self):

        nlayer     = self.config['consensus_nlayer']
        hsizes     = []
        dsizes     = []
        input_size = self.training_consensus_tensor.shape[1]
        for ilayer in range(nlayer):

            # Calculate the current hidden size
            hidden_size = int(self.config[f"consensus_hsize_{ilayer+1}"]*input_size)

            # Ensure a minimum size of the output neurons
            if hidden_size < self.output_size:
                hidden_size = self.output_size 

            # Calculate the dropout size
            if hidden_size > self.output_size:
                drop_size = float(self.config[f"consensus_drop_{ilayer+1}"])
            else:
                drop_size = 0

            # Store the info about the hidden layer network
            hsizes.append(hidden_size)
            dsizes.append(drop_size)
        self.consensus_dict = {'hidden':hsizes,'dropout':dsizes,'activation':self.config['consensus_activation'],
                               'normalization':self.config['consensus_normalization'],'norm_order':self.config['consensus_norm_order']}

    def create_consensus_datasets(self):

        # Make the datasets
        self.consensus_train_tensor_dataset = TensorDataset(self.training_consensus_tensor.detach(),self.training_consensus_tensor_targets.detach())
        self.consensus_test_tensor_dataset = TensorDataset(self.training_consensus_tensor.detach(),self.training_consensus_tensor_targets.detach())

        # Manage the dataloading
        self.consensus_train_loader = DataLoader(self.consensus_train_tensor_dataset, batch_size=int(self.config['consensus_batchsize']), shuffle=True)
        self.consensus_test_loader  = DataLoader(self.consensus_test_tensor_dataset, batch_size=int(self.config['consensus_batchsize']), shuffle=True)

    def make_consensus_model(self):
        """
        Create the model object pointers.
        """

        input_size           = self.training_consensus_tensor.shape[1]
        self.consensus_model = ConsensusNetwork(self.consensus_dict,input_size,self.output_size)
 
    def make_consensus_criterion(self):
        """
        Make the criterion objects.
        """

        self.consensus_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def make_consensus_optimizer(self):
        """
        Make the optimizer objects.
        """

        self.consensus_optimizer = optim.Adam(self.consensus_model.parameters(), lr=self.config['lr'])

    def run_consensus_model(self,checkpoint_path):

        # If we were passed a checkpoint, load it now. Otherwise, train the combination layer
        if checkpoint_path != None:
            checkpoint = torch.load(checkpoint_path)
            self.consensus_model.load_state_dict(checkpoint['consensus_model'])
            self.consensus_optimizer.load_state_dict(checkpoint['consensus_optimizer'])
        else:
            # Train the combination model
            for epoch in tqdm(range(self.config['consensus_nepoch']), total=self.config['consensus_nepoch'], disable=self.raytuning):
                self.consensus_model.train()
                loss_list = []
                for ibatch in self.consensus_train_loader:

                    # Kick off the consensus handler
                    self.consensus_optimizer.zero_grad()

                    # Unpack the batch
                    batchtensor = ibatch[0]
                    labels      = ibatch[1]
                    
                    # get the output for the current batch
                    outputs = self.consensus_model(batchtensor)
                    conloss = self.consensus_criterion(outputs, labels)
                    loss_list.append(conloss.detach().item())
                    conloss.backward()
                    self.consensus_optimizer.step()
                self.con_loss.append(loss_list)

        # Evaluate the consensus model
        self.consensus_model.eval()

        # Get the clip level predictions
        train_outputs = self.consensus_model(self.training_consensus_tensor)
        test_outputs  = self.consensus_model(self.testing_consensus_tensor)
        
        # Measure the accuracy)
        train_acc, train_auc,y_pred = self.get_acc_auc(train_outputs,self.training_consensus_tensor_targets.detach().numpy())
        test_acc, test_auc,_        = self.get_acc_auc(test_outputs,self.testing_consensus_tensor_targets.detach().numpy())

        # Make a checkpoint for RAY tuning
        if self.raytuning:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                outdict    = {'combine_model': self.combine_model.state_dict(),'combine_optimizer': self.combine_optimizer.state_dict(),
                              'consensus_model': self.consensus_model.state_dict(),'consensus_optimizer': self.consensus_optimizer.state_dict(),
                              'comb_loss':self.comb_loss,'con_loss':self.con_loss, 'config':self.config}
                torch.save(outdict,os.path.join(temp_checkpoint_dir, "full_model.pth"))
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                # Send the current training result back to Tune
                train.report({"Train_AUC": train_auc,"Train_ACC":train_acc, "Test_AUC":test_auc, "Test_ACC":test_acc,
                            "Train_AUC_clip": self.train_auc_clip,"Train_ACC_clip":self.train_acc_clip,
                            "Test_AUC_clip":self.test_auc_clip, "Test_ACC_clip":self.test_acc_clip}, checkpoint=checkpoint)
        elif not self.raytuning:
            print(f"Training Accuracy (Patient): {train_acc:0.3f}")
            print(f"Training AUC      (Patient): {train_auc:0.3f}")
            print(f"Testing Accuracy  (Patient): {test_acc:0.3f}")
            print(f"Testing AUC       (Patient): {test_auc:0.3f}")

    def update_tensors_w_probs(self):
        # Return updated dataframe with the probs
        self.train_transformed['Epilepsy_Prob'] = self.clip_training_predictions_array[:,0]
        self.train_transformed['PNES_Prob']     = self.clip_training_predictions_array[:,1]

def train_pnes_handler(config,DL_object,patient_level=False,raytuning=True,checkpoint_path=None):
    """
    Function that manages the workflow for the MLP model.
    """

    TP = train_pnes(config,DL_object,patient_level,raytuning,checkpoint_path)
    TP.run_basemodel_pipeline()

def update_pnes_handler(config,DL_object,patient_level=True,raytuning=False,checkpoint_path=None):

    if checkpoint_path==None:
        raise FileNotFoundError("Please provide a model snapshot for semi-supervised learning of new patients.")

    TP = train_pnes(config,DL_object,patient_level,raytuning,checkpoint_path)
    TP.update_basemodel_pipeline()
    
class tuning_manager:

    def __init__(self, DL_object, ncpu, ntrial, outfile, raydir, hotconfig, patient_level):
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
        self.patient_level     = patient_level

        # Configure some ray tuning info
        ray.init(num_cpus=ncpu)
        self.resources = {"cpu": 1,"gpu": 0}        

    def make_tuning_config_mlp(self,granularity='shallow'):
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

        if granularity == 'shallow':
            # Define the block specific options
            for iblock in self.subnetwork_list:

                # Hidden size selection methods. Currently limiting the max number of layers to three
                self.config[f"{iblock}_nlayer"]  = tune.randint(0, 4)
                self.config[f"{iblock}_hsize_1"] = tune.quniform(0.05, 1.5, .01)
                self.config[f"{iblock}_hsize_2"] = tune.quniform(0.05, 1.5, .01)
                self.config[f"{iblock}_hsize_3"] = tune.quniform(0.05, 1.5, .01)

                # Drouput fraction selection methods. Currently limiting the max number of layers to three (pairs to the hidden size networks)
                self.config[f"{iblock}_drop_1"] = tune.quniform(0.0, 0.5, .01)
                self.config[f"{iblock}_drop_2"] = tune.quniform(0.0, 0.5, .01)
                self.config[f"{iblock}_drop_3"] = tune.quniform(0.0, 0.5, .01)

                # Set the activation, normalization, and layer orders
                self.config[f"{iblock}_activation"]   = tune.choice(['relu','tanh','sigmoid','softmax','lrelu','pass'])
                self.config[f"{iblock}_normalization"] = tune.choice(['batch1d','pass'])
                self.config[f"{iblock}_norm_order"]   = tune.choice(['before','after'])

            # Define the combination configuration block
            self.config[f"combined_nlayer"]       = tune.randint(1, 4)
            self.config[f"combined_hsize_1"]      = tune.quniform(0.05, 1.5, .01)
            self.config[f"combined_hsize_2"]      = tune.quniform(0.05, 1.5, .01)
            self.config[f"combined_hsize_3"]      = tune.quniform(0.05, 1.5, .01)
            self.config[f"combined_drop_1"]       = tune.quniform(0.05, .5, .01)
            self.config[f"combined_drop_2"]       = tune.quniform(0.05, .5, .01)
            self.config[f"combined_drop_3"]       = tune.quniform(0.05, .5, .01)
            self.config[f"combined_activation"]   = tune.choice(['relu','tanh','sigmoid','softmax','lrelu','pass'])
            self.config[f"combined_normalization"] = tune.choice(['batch1d','pass'])
            self.config[f"combined_norm_order"]   = tune.choice(['before','after'])

            # Global fitting criteria selection
            self.config["combination_nepoch"] = tune.randint(1,101)
            self.config["consensus_nepoch"]   = tune.randint(1,101)
            self.config['lr']                 = tune.loguniform(1e-5,1e-3)
            self.config['batchsize']          = tune.quniform(32, 1024, 32)
            self.config['weight']             = tune.uniform(.1,20)

            # Consensus configuration
            self.config["consensus_batchsize"] = tune.quniform(8, 256, 8)
            self.config[f"consensus_nlayer"]   = tune.randint(1, 4)
            self.config[f"consensus_hsize_1"]  = tune.quniform(0.05, 1.5, .01)
            self.config[f"consensus_hsize_2"]  = tune.quniform(0.05, 1.5, .01)
            self.config[f"consensus_hsize_3"]  = tune.quniform(0.05, 1.5, .01)
            self.config[f"consensus_drop_1"]   = tune.quniform(0.05, .5, .01)
            self.config[f"consensus_drop_2"]   = tune.quniform(0.05, .5, .01)
            self.config[f"consensus_drop_3"]   = tune.quniform(0.05, .5, .01)
            self.config['consensus_theshold_method']             = tune.choice(['posterior','quantile'])
            self.config["consensus_theshold_yasa_prediction_00"] = tune.quniform(0.01, 1.0, .01)
            self.config["consensus_theshold_yasa_prediction_01"] = tune.quniform(0.01, 1.0, .01)
            self.config["consensus_theshold_yasa_prediction_02"] = tune.quniform(0.01, 1.0, .01)
            self.config[f"consensus_activation"]   = tune.choice(['relu','tanh','sigmoid','softmax','lrelu','pass'])
            self.config[f"consensus_normalization"] = tune.choice(['batch1d','pass'])
            self.config[f"consensus_norm_order"]   = tune.choice(['before','after'])

    def run_ray_tune_mlp(self,coldstart=False,nlayer_guess=1,h1guess=1.0,h2guess=1.0,h3guess=1.0,drop1guess=0.4,drop2guess=0.4,drop3guess=0.2,batchguess=64,lrguess=5e-5):
        
        # Define the starting parameters for the global parameters
        current_best_params = [self.hotconfig]

        # Define the search parameters
        hyperopt_search = HyperOptSearch(metric="Test_AUC", mode="max", points_to_evaluate=current_best_params, random_state_seed=42)

        # Set the number of cpus to use
        trainable_with_resources = tune.with_resources(train_pnes_handler, {"cpu": 0.5})
        trainable_with_parameters = tune.with_parameters(trainable_with_resources, DL_object=(self.DL_object), patient_level=(self.patient_level))

        # Create the tranable object
        tuner = tune.Tuner(trainable_with_parameters,param_space=self.config,
                           tune_config=tune.TuneConfig(num_samples=self.ntrial, search_alg=hyperopt_search),
                           run_config=RunConfig(storage_path=self.raydir, name="pnes_experiment",verbose=1,
                                                failure_config=train.FailureConfig(fail_fast=False)))

        # Get the hyper parameter search results
        try:
            results = tuner.fit()

            # Save the tuned results
            result_DF = results.get_dataframe()
            result_DF.to_csv(self.outfile)
        except ValueError as e:

            # Check for the most common error, mismatch of paramter space and guess space
            evaluate_keys = np.array(list(self.hotconfig.keys()))
            raykeys       = np.array(list(self.config.keys()))

            # Find any excluded keys
            print(np.setdiff1d(evaluate_keys,raykeys))
            print(np.setdiff1d(raykeys,evaluate_keys))
