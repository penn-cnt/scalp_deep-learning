import os
import pickle
import numpy as np
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


class SubNetwork_bandpower(nn.Module):
    def __init__(self, input_size_bandpower, hidden_size_bandpower, dropout_rate, normorder='first',activation='relu'):
        super(SubNetwork_bandpower, self).__init__()
        self.fc      = nn.Linear(input_size_bandpower, hidden_size_bandpower)
        self.relu    = nn.ReLU()
        self.tanh    = nn.Tanh()
        self.bn      = nn.BatchNorm1d(hidden_size_bandpower)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Handle selection of activation layer
        self.normorder = normorder
        if activation == 'relu':
            self.activation_layer = self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh
    
    def forward(self, x):
        x = self.fc(x)

        if self.normorder == 'first':
            x = self.bn(x)
            x = self.activation_layer(x)
        else:
            x = self.activation_layer(x)
            x = self.bn(x)   

        x = self.dropout(x)
        return x
    
# Make the single layered timeseries network
class SubNetwork_timeseries(nn.Module):
    def __init__(self, input_size_timeseries, hidden_size_timeseries, dropout_rate, normorder='first',activation='relu'):
        super(SubNetwork_timeseries, self).__init__()
        self.fc      = nn.Linear(input_size_timeseries, hidden_size_timeseries)
        self.relu    = nn.ReLU()
        self.tanh    = nn.Tanh()
        self.bn      = nn.BatchNorm1d(hidden_size_timeseries)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Handle selection of activation layer
        self.normorder = normorder
        if activation == 'relu':
            self.activation_layer = self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh

    def forward(self, x):
        x = self.fc(x)

        if self.normorder == 'first':
            x = self.bn(x)
            x = self.activation_layer(x)
        else:
            x = self.activation_layer(x)
            x = self.bn(x)   

        x = self.dropout(x)
        return x
    
# Make the single layered timeseries network
class SubNetwork_categorical(nn.Module):
    def __init__(self, input_size_categorical, hidden_size_categorical, dropout_rate, normorder='first',activation='relu'):
        super(SubNetwork_categorical, self).__init__()
        self.fc      = nn.Linear(input_size_categorical, hidden_size_categorical)
        self.relu    = nn.ReLU()
        self.tanh    = nn.Tanh()
        self.bn      = nn.BatchNorm1d(hidden_size_categorical)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Handle selection of activation layer
        self.normorder = normorder
        if activation == 'relu':
            self.activation_layer = self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh
    
    def forward(self, x):
        x = self.fc(x)

        if self.normorder == 'first':
            x = self.bn(x)
            x = self.activation_layer(x)
        else:
            x = self.activation_layer(x)
            x = self.bn(x)   

        x = self.dropout(x)
        return x

# Make the single layered combined network
class CombinedNetwork(nn.Module):
    def __init__(self, input_sizes, subnet_hidden_sizes, combined_subnet_size, combined_hidden_size, dropouts, output_size, normorder='first',activation='relu'):
        super(CombinedNetwork, self).__init__()
        
        self.subnet1 = SubNetwork_bandpower(input_sizes[0], subnet_hidden_sizes[0], dropout_rate=dropouts[0])
        self.subnet2 = SubNetwork_timeseries(input_sizes[1], subnet_hidden_sizes[1], dropout_rate=dropouts[1])
        self.subnet3 = SubNetwork_categorical(input_sizes[2], subnet_hidden_sizes[2], dropout_rate=dropouts[2])
        
        # Final layers after combining the sub-networks
        self.fc_combined = nn.Linear(combined_subnet_size, combined_hidden_size)
        self.relu        = nn.ReLU()
        self.tanh        = nn.Tanh()
        self.bn          = nn.BatchNorm1d(combined_hidden_size)
        self.fc_output   = nn.Linear(combined_hidden_size, output_size)
        self.dropout     = nn.Dropout(p=dropouts[3])
        self.softmax     = nn.Softmax(dim=1)
        self.sigmoid     = nn.Sigmoid()

        # Handle selection of activation layer
        self.normorder = normorder
        if activation == 'relu':
            self.activation_layer = self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh

    def forward(self, x1, x2, x3):
        out1 = self.subnet1(x1)
        out2 = self.subnet2(x2)
        out3 = self.subnet3(x3)
        
        # Concatenate the outputs of the three sub-networks
        combined = torch.cat((out1, out2, out3), dim=1)
        
        # Pass the combined output through the final layers
        combined = self.fc_combined(combined)

        if self.normorder == 'first':
            combined = self.bn(combined)
            combined = self.activation_layer(combined)
        else:
            combined = self.activation_layer(combined)        
            combined = self.bn(combined)
        
        output   = self.fc_output(combined)
        output   = self.dropout(output)
        output   = self.sigmoid(output)
        
        return output
 
def train_pnes(config,directload=False):

    torch.manual_seed(42)

    # Store the inputs
    data_objects        = pickle.load(open(config['infile'],'rb'))
    X_train_bandpower   = data_objects[0]
    X_test_bandpower    = data_objects[1]
    X_train_timeseries  = data_objects[2]
    X_test_timeseries   = data_objects[3]
    X_train_categorical = data_objects[4]
    X_test_categorical  = data_objects[5]
    Y_train             = data_objects[6]
    Y_test              = data_objects[7]
    model_block         = data_objects[8]

    # Convert NumPy arrays to PyTorch tensors
    X_train_tensor_bandpower   = torch.tensor(X_train_bandpower, dtype=torch.float32)
    X_test_tensor_bandpower    = torch.tensor(X_test_bandpower, dtype=torch.float32)
    X_train_tensor_timeseries  = torch.tensor(X_train_timeseries, dtype=torch.float32)
    X_test_tensor_timeseries   = torch.tensor(X_test_timeseries, dtype=torch.float32)
    X_train_tensor_categorical = torch.tensor(X_train_categorical, dtype=torch.float32)
    X_test_tensor_categorical  = torch.tensor(X_test_categorical, dtype=torch.float32)
    y_train_tensor             = torch.tensor(Y_train, dtype=torch.float32)
    y_test_tensor              = torch.tensor(Y_test, dtype=torch.float32)

    # Make the dataset objects
    train_dataset = TensorDataset(X_train_tensor_bandpower,X_train_tensor_timeseries,X_train_tensor_categorical, y_train_tensor)
    train_loader  = DataLoader(train_dataset, batch_size=config['batchsize'], shuffle=True)

    # Define the input sizes for the different blocks
    bandsize  = len(model_block['bandpower'])
    timesize  = len(model_block['timeseries'])
    catsize   = len(model_block['categoricals'])

    # Individual hidden arrays
    hidden_band_array = int(bandsize*config['hsize1'])
    hidden_time_array = int(timesize*config['hsize2'])
    hidden_cat_array  = int(catsize*config['hsize3'])
    combined_subnet   = hidden_band_array+hidden_time_array+hidden_cat_array
    combined_hidden   = int(combined_subnet*config['hsize_comb'])

    # Make the dropout array
    dropouts = [config['drop1'],config['drop2'],config['drop3'],config['drop_comb']]

    # Create the network sizes
    input_sizes              = [bandsize,timesize,catsize]
    subnet_hidden_sizes      = [hidden_band_array,hidden_time_array,hidden_cat_array]
    output_size              = 1

    # Make the model
    model = CombinedNetwork(input_sizes, subnet_hidden_sizes, config['hsize_comb'], dropouts, output_size, normorder=config['normorder'], activation=config['activation'])

    # Define the loss criterion
    criterion = nn.BCELoss()

    # Select the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        for inputs_bandpower, inputs_timeseries, inputs_categorical, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_bandpower,inputs_timeseries,inputs_categorical)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():

            # Get the train AUC
            y_pred      = model(X_train_tensor_bandpower,X_train_tensor_timeseries,X_train_tensor_categorical).squeeze().numpy()
            y_pred_clean = np.round(y_pred)
            y_meas_clean = Y_train
            train_auc = roc_auc_score(y_meas_clean,y_pred_clean)
            
            # Get the test AUC
            y_pred      = model(X_test_tensor_bandpower,X_test_tensor_timeseries,X_test_tensor_categorical).squeeze().numpy()
            y_pred_clean = np.round(y_pred)
            y_meas_clean = Y_test
            test_auc = roc_auc_score(y_meas_clean,y_pred_clean)

        if not directload:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                if (epoch + 1) % 5 == 0:
                    # This saves the model to the trial directory
                    torch.save(
                        model.state_dict(),
                        os.path.join(temp_checkpoint_dir, "model.pth")
                    )
                    checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

                # Send the current training result back to Tune
                train.report({"Train_AUC": train_auc,"Test_AUC":test_auc}, checkpoint=checkpoint)
        else:
            print(f"Epoch: {epoch}")
            print(f"Train AUC {train_auc:.3f}")
            print(f"Test AUC: {test_auc:.3f}")
            print("=========")

# UI Class
class tuned_mlp_handler:

    def __init__(self,ncpu,ntrial,logfile,raydir):
        self.ncpu    = ncpu
        self.ntrial  = ntrial
        self.logfile = logfile
        self.raydir  = raydir

    def test_set_config(self,filepath):

        # Roughly, .73 .72 on RAW
        config = {}
        config["hsize1"]     = 0.5
        config["hsize2"]     = 0.5
        config["hsize3"]     = 1
        config['hsize_comb'] = 0.5
        config["drop1"]      = 0.4
        config["drop2"]      = 0.4
        config["drop3"]      = 0.0
        config["drop_comb"]  = 0.4
        config['lr']         = 1e-3
        config['batchsize']  = 64
        config['normorder']  = 'after'
        config['activation'] = 'relu'
        config['infile']     = filepath

        train_pnes(config,directload=True)

    def create_config(self,filepath):
        
        # Make the config object
        self.config = {}

        # Add configuration options
        self.config["hsize1"]     = tune.uniform(0.05, 1.5)
        self.config["hsize2"]     = tune.uniform(0.05, 1.5)
        self.config["hsize3"]     = tune.uniform(0.3, 1.5)
        self.config['hsize_comb'] = tune.uniform(0.05, 1.5)
        self.config["drop1"]      = tune.quniform(0.05, .5, .05)
        self.config["drop2"]      = tune.quniform(0.05, .5, .05)
        self.config["drop3"]      = tune.quniform(0.05, .5, .05)
        self.config["drop_comb"]  = tune.quniform(0.05, .5, .05)
        self.config['lr']         = tune.loguniform(1e-5,1e-3)
        self.config['batchsize']  = tune.choice([32,64,128,256])
        self.config['normorder']  = tune.choice(['before','after'])
        self.config['activation'] = tune.choice(['relu','tanh'])
        self.config['infile']     = tune.choice([filepath])

    def run_ray_tune(self,filepath,h1guess=1.0,h2guess=1.0,h3guess=1.0,hcombguess=1.0,drop1guess=0.4,drop2guess=0.4,drop3guess=0.2,dcombguess=0.4,batchguess=64,lrguess=5e-5):
        
        # Define the search algorithm
        current_best_params = [{'hsize1': h1guess,
                                'hsize2': h2guess,
                                'hsize3': h3guess,
                                'hsize_comb': hcombguess,
                                'drop1': drop1guess,
                                'drop2': drop2guess,
                                'drop3': drop3guess,
                                'drop_comb': dcombguess,
                                'lr':lrguess,
                                'batchsize':batchguess,
                                'normorder':'before',
                                'activation': "relu",
                                'infile': filepath}]
        hyperopt_search = HyperOptSearch(metric="Train_AUC", mode="max",points_to_evaluate=current_best_params)

        # Set the number of cpus to use
        trainable_with_resources = tune.with_resources(train_pnes, {"cpu": self.ncpu})
        
        # Create the tranable object
        tuner = tune.Tuner(trainable_with_resources,param_space=self.config,tune_config=tune.TuneConfig(num_samples=self.ntrial,search_alg=hyperopt_search),run_config=RunConfig(storage_path=self.raydir, name="pnes_experiment"))

        # Get the hyper parameter search results
        results   = tuner.fit()
        result_DF = results.get_dataframe()
        result_DF.to_csv(self.logfile)