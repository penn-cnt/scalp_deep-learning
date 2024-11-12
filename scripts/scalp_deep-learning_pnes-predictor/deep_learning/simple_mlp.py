import numpy as np
from sklearn.metrics import roc_auc_score

# Torch loaders
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    
class mlp_handler:

    def __init__(self, in_object):

        self.X_train_bandpower   = in_object[0]
        self.X_test_bandpower    = in_object[1]
        self.X_train_timeseries  = in_object[2]
        self.X_test_timeseries   = in_object[3]
        self.X_train_categorical = in_object[4]
        self.X_test_categorical  = in_object[5]
        self.Y_train             = in_object[6]
        self.Y_test              = in_object[7]
        self.model_block         = in_object[8]

    def run_mlp(self, learning_rate, bsize, hidden_coefficients, lossfnc='BCE',verbose=False):

        torch.manual_seed(42)
    
        # Define the input sizes for the different blocks
        bandsize  = len(self.model_block['bandpower'])
        timesize  = len(self.model_block['timeseries'])
        catsize   = len(self.model_block['categoricals'])

        # Individual hidden arrays
        hidden_band_array = int(bandsize*hidden_coefficients[0])
        hidden_time_array = int(timesize*hidden_coefficients[1])
        hidden_cat_array  = int(catsize*hidden_coefficients[2])
        combined_subnet   = hidden_band_array+hidden_time_array+hidden_cat_array
        combined_hidden   = int(combined_subnet*hidden_coefficients[3])

        # Create the network sizes
        input_sizes              = [bandsize,timesize,catsize]
        subnet_hidden_sizes      = [hidden_band_array,hidden_time_array,hidden_cat_array]
        output_size              = 1

        # Define the model
        model = CombinedNetwork(input_sizes, subnet_hidden_sizes, combined_subnet, combined_hidden, output_size)

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
                print(f"Test AUC Score: {test_auc:.2f}")
        return train_auc,test_auc