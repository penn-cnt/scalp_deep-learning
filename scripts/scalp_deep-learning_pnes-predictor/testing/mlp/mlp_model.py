import numpy as np
import pandas as PD
from sys import exit
from tqdm import tqdm
from itertools import product
from scipy.stats import zscore
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,GroupShuffleSplit 
from sklearn.preprocessing import StandardScaler,LabelBinarizer,PowerTransformer

# Torch loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

########################################################
####### Variable hidden network layer functions. #######
########################################################

# Make the variable layered bandpower network
class SubNetwork_bandpower_multi(nn.Module):
    def __init__(self, input_size_bandpower, hidden_size_bandpower, dropout_rate):
        super(SubNetwork_bandpower_multi, self).__init__()
        
        # Make a dictionary of layers so we can abstract the number of hidden layers for hyper paramter searches
        self.fc      = {}
        self.relu    = {}
        self.bn      = {}
        self.dropout = {}

        # Loop over the number of hidden layers and create the objects to propagate data
        for idx,output_shape in enumerate(hidden_size_bandpower):
            
            # Get the input shape
            if idx == 0:
                last_size = input_size_bandpower

            # Store the network for this layer
            self.fc[idx]      = nn.Linear(last_size, output_shape)
            self.bn[idx]      = nn.BatchNorm1d(output_shape)
            self.relu[idx]    = nn.ReLU()
            self.dropout[idx] = nn.Dropout(p=dropout_rate[idx])

            # Update the last_size object for next loop through
            last_size = output_shape
    
    def forward(self, x):

        # Loop over the layers of the hidden network
        for ikey in self.fc.keys():
            x = self.fc[ikey](x)
            x = self.bn[ikey](x)
            x = self.relu[ikey](x)
            x = self.dropout[ikey](x)
        return x

# Make the variable layered timeseries network
class SubNetwork_timeseries_multi(nn.Module):
    def __init__(self, input_size_timeseries, hidden_size_timeseries, dropout_rate):
        super(SubNetwork_timeseries_multi, self).__init__()
        
        # Make a dictionary of layers so we can abstract the number of hidden layers for hyper paramter searches
        self.fc      = {}
        self.relu    = {}
        self.bn      = {}
        self.dropout = {}

        # Loop over the number of hidden layers and create the objects to propagate data
        for idx,output_shape in enumerate(hidden_size_timeseries):
            
            # Get the input shape
            if idx == 0:
                last_size = input_size_timeseries

            # Store the network for this layer
            self.fc[idx]      = nn.Linear(last_size, output_shape)
            self.bn[idx]      = nn.BatchNorm1d(output_shape)
            self.relu[idx]    = nn.ReLU()
            self.dropout[idx] = nn.Dropout(p=dropout_rate[idx])

            # Update the last_size object for next loop through
            last_size = output_shape
    
    def forward(self, x):

        # Loop over the layers of the hidden network
        for ikey in self.fc.keys():
            x = self.fc[ikey](x)
            x = self.bn[ikey](x)
            x = self.relu[ikey](x)
            x = self.dropout[ikey](x)
        return x

# Make the variable layered sleepstate network
class SubNetwork_categorical_multi(nn.Module):
    def __init__(self, input_size_categorical, hidden_size_categorical, dropout_rate):
        super(SubNetwork_categorical_multi, self).__init__()
        
        # Make a dictionary of layers so we can abstract the number of hidden layers for hyper paramter searches
        self.fc      = {}
        self.relu    = {}
        self.bn      = {}
        self.dropout = {}

        # Loop over the number of hidden layers and create the objects to propagate data
        for idx,output_shape in enumerate(hidden_size_categorical):
            
            # Get the input shape
            if idx == 0:
                last_size = input_size_categorical

            # Store the network for this layer
            self.fc[idx]      = nn.Linear(last_size, output_shape)
            self.bn[idx]      = nn.BatchNorm1d(output_shape)
            self.relu[idx]    = nn.ReLU()
            self.dropout[idx] = nn.Dropout(p=dropout_rate[idx])

            # Update the last_size object for next loop through
            last_size = output_shape
    
    def forward(self, x):

        # Loop over the layers of the hidden network
        for ikey in self.fc.keys():
            x = self.fc[ikey](x)
            x = self.bn[ikey](x)
            x = self.relu[ikey](x)
            x = self.dropout[ikey](x)
        return x

# Make the single layered combined network
class CombinedNetwork_multi(nn.Module):
    def __init__(self, input_sizes, subnet_hidden_sizes, dropout_rates, output_size):
        super(CombinedNetwork_multi, self).__init__()
        
        self.subnet1 = SubNetwork_bandpower_multi(input_sizes[0], subnet_hidden_sizes[0], dropout_rate=dropout_rates[0])
        self.subnet2 = SubNetwork_timeseries_multi(input_sizes[1], subnet_hidden_sizes[1], dropout_rate=dropout_rates[1])
        self.subnet3 = SubNetwork_categorical_multi(input_sizes[2], subnet_hidden_sizes[2], dropout_rate=dropout_rates[2])
        
        # Get the correct input size from variable hidden layers
        combined_subnet_size = subnet_hidden_sizes[0][-1]+subnet_hidden_sizes[1][-1]+subnet_hidden_sizes[2][-1]
        combined_hidden_size = int(0.5*combined_subnet_size)

        # Final layers after combining the sub-networks
        self.fc_combined = nn.Linear(combined_subnet_size, combined_hidden_size)
        self.bn          = nn.BatchNorm1d(combined_hidden_size)
        self.relu        = nn.ReLU()
        self.fc_output   = nn.Linear(combined_hidden_size, output_size)
        self.dropout     = nn.Dropout(p=0.4)
        self.softmax     = nn.Softmax(dim=1)
        self.sigmoid     = nn.Sigmoid()
    
    def forward(self, x1, x2, x3):
        out1 = self.subnet1(x1)
        out2 = self.subnet2(x2)
        out3 = self.subnet3(x3)
        
        # Concatenate the outputs of the three sub-networks
        combined = torch.cat((out1, out2, out3), dim=1)
        
        # Pass the combined output through the final layers
        combined = self.fc_combined(combined)
        combined = self.bn(combined)
        combined = self.relu(combined)
        output   = self.fc_output(combined)
        output   = self.dropout(output)
        output   = self.sigmoid(output)
        
        return output

######################################################
####### Single hidden network layer functions. #######
######################################################
 
# Make the single layered bandpower network
class SubNetwork_bandpower(nn.Module):
    def __init__(self, input_size_bandpower, hidden_size_bandpower, dropout_rate):
        super(SubNetwork_bandpower, self).__init__()
        self.fc      = nn.Linear(input_size_bandpower, hidden_size_bandpower)
        self.relu    = nn.ReLU()
        self.bn      = nn.BatchNorm1d(hidden_size_bandpower)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x
    
# Make the single layered timeseries network
class SubNetwork_timeseries(nn.Module):
    def __init__(self, input_size_timeseries, hidden_size_timeseries, dropout_rate):
        super(SubNetwork_timeseries, self).__init__()
        self.fc      = nn.Linear(input_size_timeseries, hidden_size_timeseries)
        self.relu    = nn.ReLU()
        self.bn      = nn.BatchNorm1d(hidden_size_timeseries)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x
    
# Make the single layered timeseries network
class SubNetwork_categorical(nn.Module):
    def __init__(self, input_size_categorical, hidden_size_categorical, dropout_rate):
        super(SubNetwork_categorical, self).__init__()
        self.fc      = nn.Linear(input_size_categorical, hidden_size_categorical)
        self.relu    = nn.ReLU()
        self.bn      = nn.BatchNorm1d(hidden_size_categorical)
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

# Make the single layered combined network
class CombinedNetwork(nn.Module):
    def __init__(self, input_sizes, subnet_hidden_sizes, combined_subnet_size, combined_hidden_size, output_size):
        super(CombinedNetwork, self).__init__()
        
        self.subnet1 = SubNetwork_bandpower(input_sizes[0], subnet_hidden_sizes[0], dropout_rate=0.4)
        self.subnet2 = SubNetwork_timeseries(input_sizes[1], subnet_hidden_sizes[1], dropout_rate=0.4)
        self.subnet3 = SubNetwork_categorical(input_sizes[2], subnet_hidden_sizes[2], dropout_rate=0.0)
        
        # Final layers after combining the sub-networks
        self.fc_combined = nn.Linear(combined_subnet_size, combined_hidden_size)
        self.relu        = nn.ReLU()
        self.bn          = nn.BatchNorm1d(combined_hidden_size)
        self.fc_output   = nn.Linear(combined_hidden_size, output_size)
        self.dropout     = nn.Dropout(p=0.4)
        self.softmax     = nn.Softmax(dim=1)
        self.sigmoid     = nn.Sigmoid()
    
    def forward(self, x1, x2, x3):
        out1 = self.subnet1(x1)
        out2 = self.subnet2(x2)
        out3 = self.subnet3(x3)
        
        # Concatenate the outputs of the three sub-networks
        combined = torch.cat((out1, out2, out3), dim=1)
        
        # Pass the combined output through the final layers
        combined = self.fc_combined(combined)
        combined = self.relu(combined)
        combined = self.bn(combined)
        output   = self.fc_output(combined)
        output   = self.dropout(output)
        output   = self.sigmoid(output)
        
        return output

#########################################
####### Calling function for MLP. #######
#########################################

class mlp_handler:

    def __init__(self):

        pass

    def run_mlp_single(self,learning_rate,bsize,lossfnc='BCE'):

        torch.manual_seed(42)
    
        # Define the input sizes for the different blocks
        bandsize  = len(self.model_block['bandpower'])
        timesize  = len(self.model_block['timeseries'])
        catsize   = len(self.model_block['categoricals'])

        # Create the network sizes
        input_sizes              = [bandsize,timesize,catsize]
        subnet_hidden_sizes      = [int(0.5*bandsize),int(0.5*(timesize)),3]
        combined_subnet_size     = np.sum(subnet_hidden_sizes)
        combined_hidden_size     = int(0.5*combined_subnet_size)
        output_size              = 1

        # Define the model
        model = CombinedNetwork(input_sizes, subnet_hidden_sizes, combined_subnet_size, combined_hidden_size, output_size)

        # Define the criterion
        if lossfnc == 'BCE':
            criterion = nn.BCELoss()
        elif lossfnc == 'CE':
            criterion = nn.CrossEntropyLoss()
        
        # Select the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Convert NumPy arrays to PyTorch tensors
        self.X_train_tensor_bandpower   = torch.tensor(self.X_train_bandpower, dtype=torch.float32)
        self.X_test_tensor_bandpower    = torch.tensor(self.X_test_bandpower, dtype=torch.float32)
        self.X_train_tensor_timeseries  = torch.tensor(self.X_train_timeseries, dtype=torch.float32)
        self.X_test_tensor_timeseries   = torch.tensor(self.X_test_timeseries, dtype=torch.float32)
        self.X_train_tensor_categorical = torch.tensor(self.X_train_categorical, dtype=torch.float32)
        self.X_test_tensor_categorical  = torch.tensor(self.X_test_categorical, dtype=torch.float32)
        self.y_train_tensor             = torch.tensor(self.Y_train, dtype=torch.float32)
        self.y_test_tensor              = torch.tensor(self.Y_test, dtype=torch.float32)

        # Make the dataset objects
        self.train_dataset = TensorDataset(self.X_train_tensor_bandpower,self.X_train_tensor_timeseries,self.X_train_tensor_categorical, self.y_train_tensor)
        self.test_dataset  = TensorDataset(self.X_test_tensor_bandpower,self.X_test_tensor_timeseries,self.X_test_tensor_categorical, self.y_test_tensor)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=bsize, shuffle=True)
        self.test_loader   = DataLoader(self.test_dataset, batch_size=bsize, shuffle=False)

        # Train the model
        num_epochs = 25
        loss_vals  = []
        for epoch in range(num_epochs):
            model.train()
            for inputs_bandpower, inputs_timeseries, inputs_categorical, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(inputs_bandpower,inputs_timeseries,inputs_categorical)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            loss_vals.append(loss.item())

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():

            # Get the train AUC
            y_pred      = model(self.X_train_tensor_bandpower,self.X_train_tensor_timeseries,self.X_train_tensor_categorical).squeeze().numpy()
            y_pred_clean = np.round(y_pred)
            y_meas_clean = self.Y_train
            auc = roc_auc_score(y_meas_clean,y_pred_clean)
            print(f"Train AUC Score: {auc:.2f}")

            # Get the test AUC
            y_pred      = model(self.X_test_tensor_bandpower,self.X_test_tensor_timeseries,self.X_test_tensor_categorical).squeeze().numpy()
            y_pred_clean = np.round(y_pred)
            y_meas_clean = self.Y_test
            auc = roc_auc_score(y_meas_clean,y_pred_clean)
            print(f"Test AUC Score: {auc:.2f}")

    def run_mlp_multi(self, learning_rate, bsize, hidden_coefficients, dropouts, lossfnc='BCE',verbose=False):

        torch.manual_seed(42)
    
        # Define the input sizes for the different blocks
        bandsize  = len(self.model_block['bandpower'])
        timesize  = len(self.model_block['timeseries'])
        catsize   = len(self.model_block['categoricals'])

        # Individual hidden arrays
        hidden_band_array = [int(bandsize*ival) for ival in hidden_coefficients[0]]
        hidden_time_array = [int(timesize*ival) for ival in hidden_coefficients[1]]
        hidden_cat_array  = [int(catsize*ival) for ival in hidden_coefficients[2]]

        # Create the network sizes
        input_sizes              = [bandsize,timesize,catsize]
        subnet_hidden_sizes      = [hidden_band_array,hidden_time_array,hidden_cat_array]
        output_size              = 1

        # Define the model
        model = CombinedNetwork_multi(input_sizes, subnet_hidden_sizes, dropouts, output_size)

        # Define the criterion
        if lossfnc == 'BCE':
            criterion = nn.BCELoss()
        elif lossfnc == 'CE':
            criterion = nn.CrossEntropyLoss()
        
        # Select the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Convert NumPy arrays to PyTorch tensors
        self.X_train_tensor_bandpower   = torch.tensor(self.X_train_bandpower, dtype=torch.float32)
        self.X_test_tensor_bandpower    = torch.tensor(self.X_test_bandpower, dtype=torch.float32)
        self.X_train_tensor_timeseries  = torch.tensor(self.X_train_timeseries, dtype=torch.float32)
        self.X_test_tensor_timeseries   = torch.tensor(self.X_test_timeseries, dtype=torch.float32)
        self.X_train_tensor_categorical = torch.tensor(self.X_train_categorical, dtype=torch.float32)
        self.X_test_tensor_categorical  = torch.tensor(self.X_test_categorical, dtype=torch.float32)
        self.y_train_tensor             = torch.tensor(self.Y_train, dtype=torch.float32)
        self.y_test_tensor              = torch.tensor(self.Y_test, dtype=torch.float32)

        # Make the dataset objects
        self.train_dataset = TensorDataset(self.X_train_tensor_bandpower,self.X_train_tensor_timeseries,self.X_train_tensor_categorical, self.y_train_tensor)
        self.test_dataset  = TensorDataset(self.X_test_tensor_bandpower,self.X_test_tensor_timeseries,self.X_test_tensor_categorical, self.y_test_tensor)
        self.train_loader  = DataLoader(self.train_dataset, batch_size=bsize, shuffle=True)
        self.test_loader   = DataLoader(self.test_dataset, batch_size=bsize, shuffle=False)

        # Train the model
        num_epochs = 2
        loss_vals  = []
        for epoch in range(num_epochs):
            model.train()
            for inputs_bandpower, inputs_timeseries, inputs_categorical, labels in self.train_loader:
                optimizer.zero_grad()
                outputs = model(inputs_bandpower,inputs_timeseries,inputs_categorical)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            if verbose:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            loss_vals.append(loss.item())

        # Evaluate the model on the test set
        model.eval()
        with torch.no_grad():

            # Get the train AUC
            y_pred      = model(self.X_train_tensor_bandpower,self.X_train_tensor_timeseries,self.X_train_tensor_categorical).squeeze().numpy()
            y_pred_clean = np.round(y_pred)
            y_meas_clean = self.Y_train
            train_auc = roc_auc_score(y_meas_clean,y_pred_clean)
            
            # Get the test AUC
            y_pred      = model(self.X_test_tensor_bandpower,self.X_test_tensor_timeseries,self.X_test_tensor_categorical).squeeze().numpy()
            y_pred_clean = np.round(y_pred)
            y_meas_clean = self.Y_test
            test_auc = roc_auc_score(y_meas_clean,y_pred_clean)
            
            if verbose:
                print(f"Train AUC Score: {train_auc:.2f}")
                print(f"Test AUC Score: {auc:.2f}")
        return train_auc,test_auc

##########################################################
####### Functions for preparing data for modeling. #######
##########################################################

class mlp_prep(mlp_handler):

    def __init__(self,indata):
        
        # Store the data to the class instance
        self.indata = indata

    def processing_pipeline(self):
        self.map_targets()
        self.drop_extra_cols()
        self.define_column_groups()
        self.make_model_groups()
        self.data_rejection()
        self.apply_column_transform()
        self.sleep_to_categorical()
        self.target_to_categorical()
        self.define_inputs()

    def multiprocessing_handler(self):
        pass

    def multiprocess_grid(self):

        # Make the base hyper parameter grid for timeseries and bandpower
        hbase_large_1d = 0.2*(1+np.arange(10))
        hbase_large_1d = list(np.round(hbase_large_1d,1))

        # Make the 2-layer options
        foo = product(hbase_large_1d,hbase_large_1d)
        hbase_large_2d = [list(x) for x in foo]

        # Make the combined large grid
        hbase_large_combined = hbase_large_1d.copy()
        hbase_large_combined.extend(hbase_large_2d)

        # Make the base hyper parameter grid for categorical
        hbase_small_1d = [2,1,0.5]

        # Make the 2-layer options
        foo = product(hbase_small_1d,hbase_small_1d)
        hbase_small_2d = [list(x) for x in foo]

        # Make the combined small grid
        hbase_small_combined = hbase_small_1d.copy()
        hbase_small_combined.extend(hbase_small_2d)

        # Get the full hidden network
        foo = product(hbase_large_combined,hbase_large_combined,hbase_small_combined)
        hidden_network_inputs = []
        for x in foo:
            tmp = []
            for y in x:
                if type(y) != list:
                    tmp.append([y])
                else:
                    tmp.append(y)
            hidden_network_inputs.append(tmp)

        # Make the drop out network
        drop_val       = 0.3
        dropout_inputs = []
        for irow in hidden_network_inputs:
            jrow=irow.copy()
            for netind,inet in enumerate(irow):
                for valind,ival in enumerate(inet):
                    jrow[netind][valind]=drop_val
            dropout_inputs.append(jrow)

    def call_mlp_multiprocess(self,inputs,semaphore):

        # Break out the inputs
        outpath = inputs[0]
        ilr     = inputs[1]
        ibatch  = inputs[2]
        igrid   = inputs[3]
        idrop   = inputs[4]

        # Define output columns
        outcols = ['learning_rate','batch_size','bandpower_network','timeseries_network','categorical_network',
                   'bandpower_dropout','timeseries_drouput','categorical_dropout','train_auc','test_auc']
        
        # Run the model
        train_auc,test_auc = mlp_handler.run_mlp_multi(self,ilr,ibatch,igrid,idrop)

        # Store results
        hidden_band_str = '|'.join(str(x) for x in igrid[0])
        hidden_time_str = '|'.join(str(x) for x in igrid[1])
        hidden_cat_str  = '|'.join(str(x) for x in igrid[2])
        drop_band_str   = '|'.join(str(x) for x in idrop[0])
        drop_time_str   = '|'.join(str(x) for x in idrop[1])
        drop_cat_str    = '|'.join(str(x) for x in idrop[2])
        iarr  = np.array([ilr,ibatch,hidden_band_str,hidden_time_str,hidden_cat_str,drop_band_str,drop_time_str,drop_cat_str,train_auc,test_auc]).reshape((1,-1))
        iDF   = PD.DataFrame(iarr,columns=outcols)

        # Append to output file
        with semaphore:        
            outdf = PD.read_csv(outpath)
            outdf = PD.concat((outdf,iDF)).reset_index(drop=True)
            outdf.to_csv(outpath,index=False)

    def call_mlp(self):

        # Make a hyper parameter dataframe to store results
        outDF = PD.DataFrame(columns=['learning_rate','batch_size','bandpower_network','timeseries_network','categorical_network',
                                      'bandpower_dropout','timeseries_drouput','categorical_dropout','train_auc','test_auc'])

        # Define the hyperparameters
        learning_rate_grid = [1e-5,5e-5,1e-4,5e-4,1e-3]
        batch_size_grid    = [32,64,96,128,160,192,224,256]
        hidden_grid        = [[[0.75],[0.75],[3]]]
        dropout_grid       = [[[0.4],[0.4],[0]]]
        for ilr in learning_rate_grid:
            for ibatch in batch_size_grid:
                for igrid in hidden_grid:
                    for idrop in dropout_grid:

                        # Run the model
                        train_auc,test_auc = mlp_handler.run_mlp_multi(self,ilr,ibatch,igrid,idrop)

                        # Store results
                        hidden_band_str = '|'.join(str(x) for x in igrid[0])
                        hidden_time_str = '|'.join(str(x) for x in igrid[1])
                        hidden_cat_str  = '|'.join(str(x) for x in igrid[2])
                        drop_band_str   = '|'.join(str(x) for x in idrop[0])
                        drop_time_str   = '|'.join(str(x) for x in idrop[1])
                        drop_cat_str    = '|'.join(str(x) for x in idrop[2])
                        iarr  = np.array([ilr,ibatch,hidden_band_str,hidden_time_str,hidden_cat_str,drop_band_str,drop_time_str,drop_cat_str,train_auc,test_auc]).reshape((1,-1))
                        iDF   = PD.DataFrame(iarr,columns=outDF.columns)
                        outDF = PD.concat((outDF,iDF))

                #mlp_handler.run_mlp_single(self,ilr,ibatch)
                print("=================")
                outDF.to_csv("hyperparameters.csv",index=False)

    def map_targets(self):
        self.tmap             = {'pnes':0,'epilepsy':1}
        self.indata['target'] = self.indata['target'].apply(lambda x:self.tmap[x])

    def drop_extra_cols(self):
        self.indata = self.indata.drop(['file','t_end','t_start','t_window'],axis=1)

    def make_model_groups(self,method='raw'):

        if method == 'raw':
            # Raw split on length
            self.indata = self.indata.drop(['uid'],axis=1)
            self.train_raw, self.test_raw = train_test_split(self.indata, test_size=0.33, random_state=42)
        elif method == 'uid':
            # Split on uid
            splitter              = GroupShuffleSplit(test_size=.33, n_splits=1, random_state = 42)
            split                 = splitter.split(self.indata, groups=self.indata['uid'])
            train_inds, test_inds = next(split)

            # Drop the uid column now
            self.indata = self.indata.drop(['uid'],axis=1)

            # get the train and test indices
            self.train_raw = self.indata.iloc[train_inds]
            self.test_raw  = self.indata.iloc[test_inds]

    def define_column_groups(self):
        """
        Define the modeling blocks based on column name.
        """

        # Store the columns into the right modeling block for preprocessing and initial fits.
        self.iso_cols        = []
        self.transform_block = {'standard_scaler_wlog10':[],'yeo-johnson':[],'passthrough':[]}
        self.model_block     = {'bandpower':[],'timeseries':[],'categoricals':[],'targets':[]}
        for icol in self.indata.columns:
            if 'spectral_energy_welch' in icol:
                self.transform_block['standard_scaler_wlog10'].append(icol)
                self.model_block['bandpower'].append(icol)
            elif 'stdev' in icol:
                self.transform_block['standard_scaler_wlog10'].append(icol)
                self.model_block['timeseries'].append(icol)
            elif 'rms' in icol:
                self.transform_block['standard_scaler_wlog10'].append(icol)
                self.model_block['timeseries'].append(icol)
            elif 'line_length' in icol:
                self.transform_block['standard_scaler_wlog10'].append(icol)
                self.model_block['timeseries'].append(icol)
            elif 'median' in icol:
                self.iso_cols.append((icol,0.05))
                self.transform_block['yeo-johnson'].append(icol)
                self.model_block['timeseries'].append(icol)
            elif 'quantile' in icol:
                self.transform_block['yeo-johnson'].append(icol)
                self.model_block['timeseries'].append(icol)
            elif 'AD' in icol:
                self.transform_block['yeo-johnson'].append(icol)
                self.model_block['timeseries'].append(icol)
            elif 'sleep' in icol:
                self.transform_block['passthrough'].append(icol)
            elif 'target' in icol:
                self.transform_block['passthrough'].append(icol)

    def data_rejection(self):

        # Alert user
        print("Running isolation forest to reject the most extreme time segments (by median).")

        # Run an isolation forest across each feature in the training block
        self.iso_forests = {}
        for ipair in self.iso_cols:
            icol                   = ipair[0]
            contam_factor          = ipair[1]
            ISO                    = IsolationForest(contamination=contam_factor, random_state=42)
            self.iso_forests[icol] = ISO.fit(self.train_raw[icol].values.reshape((-1,1)))

        # Get the training mask
        train_2d_mask = np.zeros((self.train_raw.shape[0],len(self.iso_cols)))
        for idx,ipair in enumerate(self.iso_cols):
            icol                 = ipair[0]
            train_2d_mask[:,idx] = self.iso_forests[icol].predict(self.train_raw[icol].values.reshape((-1,1)))
        train_mask     = (train_2d_mask==1).all(axis=1)
        self.train_raw = self.train_raw.iloc[train_mask]
        
        # Get the testing mask
        test_2d_mask = np.zeros((self.test_raw.shape[0],len(self.iso_cols)))
        for idx,ipair in enumerate(self.iso_cols):
            icol                = ipair[0]
            test_2d_mask[:,idx] = self.iso_forests[icol].predict(self.test_raw[icol].values.reshape((-1,1)))
        test_mask     = (test_2d_mask==1).all(axis=1)
        self.test_raw = self.test_raw.iloc[test_mask]

        print(f"Isolation Forest reduced training size to {train_mask.sum()} from {train_mask.size} samples.")
        print(f"Isolation Forest reduced test size to {test_mask.sum()} from {test_mask.size} samples.")

    def apply_column_transform(self):

        # Create the column transformation actions
        ct = ColumnTransformer([("standard_scaler_wlog10", StandardScaler(), self.transform_block['standard_scaler_wlog10']),
                                ("yeo-johnson", PowerTransformer('yeo-johnson'), self.transform_block['yeo-johnson']),
                                ("pass_encoder", 'passthrough', self.transform_block['passthrough'])])

        # Apply the needed log-transformations
        print("Applying log-transformation.")
        self.train_raw[self.transform_block['standard_scaler_wlog10']] = np.log10(self.train_raw[self.transform_block['standard_scaler_wlog10']])
        self.test_raw[self.transform_block['standard_scaler_wlog10']]  = np.log10(self.test_raw[self.transform_block['standard_scaler_wlog10']])

        # Convert the data
        print("Applying distribution scaling.")
        ct.fit(self.train_raw)
        train_transformed = ct.transform(self.train_raw)
        test_transformed  = ct.transform(self.test_raw)

        # Make a new flat column header
        flat_cols = [x for xs in ct._columns for x in xs]
        self.train_transformed = PD.DataFrame(train_transformed,columns=flat_cols)
        self.test_transformed  = PD.DataFrame(test_transformed,columns=flat_cols)

    def sleep_to_categorical(self):

        # Apply the label binarizer to the sleep labels
        LB = LabelBinarizer()
        LB.fit(self.train_transformed['sleep_state'])
        sleep_labels = [f"sleep_{int(x):02d}" for x in LB.classes_]
        self.model_block['categoricals'].extend(sleep_labels)
        
        # Add in the sleep labels to train
        sleep_vectors = LB.transform(self.train_transformed['sleep_state'])
        self.train_transformed.drop(['sleep_state'],axis=1,inplace=True)
        self.train_transformed[sleep_labels] = sleep_vectors

        # Add in the sleep labels to test
        sleep_vectors = LB.transform(self.test_transformed['sleep_state'])
        self.test_transformed.drop(['sleep_state'],axis=1,inplace=True)
        self.test_transformed[sleep_labels] = sleep_vectors

    def target_to_categorical(self,multiclass_format=False):

        if multiclass_format:
            # Apply the label binarizer to the target labels
            LB = LabelBinarizer()
            LB.fit(self.train_transformed['target'])
            self.target_labels = [f"target_{int(x):02d}" for x in LB.classes_]
            self.model_block['targets'].extend(self.target_labels)
            
            # Add in the sleep labels to train
            target_vectors = LB.transform(self.train_transformed['target'])
            target_vectors = np.hstack((target_vectors, 1 - target_vectors))
            self.train_transformed.drop(['target'],axis=1,inplace=True)
            self.train_transformed[self.target_labels] = target_vectors

            # Add in the sleep labels to test
            target_vectors = LB.transform(self.test_transformed['target'])
            target_vectors = np.hstack((target_vectors, 1 - target_vectors))
            self.test_transformed.drop(['target'],axis=1,inplace=True)
            self.test_transformed[self.target_labels] = target_vectors
        else:
            self.model_block['targets'].append('target')

    def define_inputs(self):
        self.Ycols = self.model_block['targets']
        self.Xcols = np.setdiff1d(self.train_transformed.columns,self.Ycols)
        self.X_train_bandpower   = self.train_transformed[self.model_block['bandpower']].values
        self.X_test_bandpower    = self.test_transformed[self.model_block['bandpower']].values
        self.X_train_timeseries  = self.train_transformed[self.model_block['timeseries']].values
        self.X_test_timeseries   = self.test_transformed[self.model_block['timeseries']].values
        self.X_train_categorical = self.train_transformed[self.model_block['categoricals']].values
        self.X_test_categorical  = self.test_transformed[self.model_block['categoricals']].values
        self.Y_train             = self.train_transformed[self.Ycols].values
        self.Y_test              = self.test_transformed[self.Ycols].values