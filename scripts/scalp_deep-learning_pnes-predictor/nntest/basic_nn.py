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

class CombinedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, output_size):
        super(CombinedNetwork, self).__init__()

        self.fc      = nn.Linear(input_size, hidden_size)
        self.fc_out  = nn.Linear(hidden_size, output_size)
        self.relu    = nn.ReLU()
        self.bn      = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x

def train_pnes(config,DL_object,debug=False,patient_level=False,directload=False):
    """
    Function that manages the workflow for the MLP model.
    """

    # Unpack the data for our model
    model_block       = DL_object[0]
    train_transformed = DL_object[1]
    
    # get the training columns
    cols  = np.concatenate([model_block['frequency'],model_block['time'],model_block['categorical']])
    tcols = model_block['target']

    # get the train dataset
    train_dataset = torch.from_numpy(train_transformed[cols].values.astype(np.float32))
    train_arr     = train_transformed[tcols].values.astype(np.float32)
    train_targets = torch.from_numpy(train_arr)

    # Make the datasets
    train_tensor_dataset = TensorDataset(train_dataset,train_targets)
    train_loader         = DataLoader(train_tensor_dataset, batch_size=config['batchsize'], shuffle=True)
    
    # Make the model
    model = CombinedNetwork(cols.size, int(0.5*cols.size), 0.2, 2)
    
    # Define the loss criterion
    sums              = train_targets.numpy().sum(axis=0)
    pos_weight        = 100*torch.tensor([sums[0]/sums[1]])
    patient_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Select the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    num_epochs  = 10
    model.train()
    for epoch in tqdm(range(num_epochs), total=num_epochs):

        # Kick off the consensus handler
        optimizer.zero_grad()

        # get the predicted outputs
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss    = patient_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    outputs = model(train_dataset)
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

    print(train_acc,train_auc)
    exit()