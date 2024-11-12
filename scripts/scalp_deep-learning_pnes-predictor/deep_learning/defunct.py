# Make the variable layered bandpower network
class SubNetwork_bandpower_multi(nn.Module):
    def __init__(self, input_size_bandpower, hidden_size_bandpower, dropout_rate, normorder='first',activation='relu'):
        super(SubNetwork_bandpower_multi, self).__init__()
        
        # Make a dictionary of layers so we can abstract the number of hidden layers for hyper paramter searches
        self.fc        = {}
        self.relu      = {}
        self.tanh      = {}
        self.bn        = {}
        self.dropout   = {}
        self.normorder = normorder

        # Loop over the number of hidden layers and create the objects to propagate data
        for idx,output_shape in enumerate(hidden_size_bandpower):
            
            # Get the input shape
            if idx == 0:
                last_size = input_size_bandpower

            # Store the network for this layer
            self.fc[idx]      = nn.Linear(last_size, output_shape)
            self.bn[idx]      = nn.BatchNorm1d(output_shape)
            self.relu[idx]    = nn.ReLU()
            self.tanh[idx]    = nn.Tanh()
            self.dropout[idx] = nn.Dropout(p=dropout_rate[idx])

            # Update the last_size object for next loop through
            last_size = output_shape
        
        if activation == 'relu':
            self.activation_layer = self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh
    
    def forward(self, x):

        # Loop over the layers of the hidden network
        for ikey in self.fc.keys():
            x = self.fc[ikey](x)

            if self.normorder == 'first':
                x = self.bn[ikey](x)
                x = self.activation_layer[ikey](x)
            else:
                x = self.activation_layer[ikey](x)
                x = self.bn[ikey](x)
            
            x = self.dropout[ikey](x)
        return x

# Make the variable layered timeseries network
class SubNetwork_timeseries_multi(nn.Module):
    def __init__(self, input_size_timeseries, hidden_size_timeseries, dropout_rate, normorder='first',activation='relu'):
        super(SubNetwork_timeseries_multi, self).__init__()
        
        # Make a dictionary of layers so we can abstract the number of hidden layers for hyper paramter searches
        self.fc        = {}
        self.relu      = {}
        self.bn        = {}
        self.dropout   = {}
        self.normorder = normorder

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

        if activation == 'relu':
            self.activation_layer = self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh
    
    def forward(self, x):

        # Loop over the layers of the hidden network
        for ikey in self.fc.keys():
            x = self.fc[ikey](x)

            if self.normorder == 'first':
                x = self.bn[ikey](x)
                x = self.activation_layer[ikey](x)
            else:
                x = self.activation_layer[ikey](x)
                x = self.bn[ikey](x)

            x = self.dropout[ikey](x)
        return x

# Make the variable layered sleepstate network
class SubNetwork_categorical_multi(nn.Module):
    def __init__(self, input_size_categorical, hidden_size_categorical, dropout_rate, normorder='first',activation='relu'):
        super(SubNetwork_categorical_multi, self).__init__()
        
        # Make a dictionary of layers so we can abstract the number of hidden layers for hyper paramter searches
        self.fc        = {}
        self.relu      = {}
        self.tanh      = {}
        self.bn        = {}
        self.dropout   = {}
        self.normorder = normorder

        # Loop over the number of hidden layers and create the objects to propagate data
        for idx,output_shape in enumerate(hidden_size_categorical):
            
            # Get the input shape
            if idx == 0:
                last_size = input_size_categorical

            # Store the network for this layer
            self.fc[idx]      = nn.Linear(last_size, output_shape)
            self.bn[idx]      = nn.BatchNorm1d(output_shape)
            self.relu[idx]    = nn.ReLU()
            self.tanh[idx]    = nn.Tanh()
            self.dropout[idx] = nn.Dropout(p=dropout_rate[idx])

            # Update the last_size object for next loop through
            last_size = output_shape

        if activation == 'relu':
            self.activation_layer= self.relu
        elif activation == 'tanh':
            self.activation_layer = self.tanh

    def forward(self, x):

        # Loop over the layers of the hidden network
        for ikey in self.fc.keys():
            x = self.fc[ikey](x)

            if self.normorder == 'first':
                x = self.bn[ikey](x)
                x = self.activation_layer[ikey](x)
            else:
                x = self.activation_layer[ikey](x)
                x = self.bn[ikey](x)            
            
            x = self.dropout[ikey](x)
        return x

# Make the single layered combined network
class CombinedNetwork_multi(nn.Module):
    def __init__(self, input_sizes, subnet_hidden_sizes, combined_hidden_size, dropout_rates, output_size, normorder='first',activation='relu'):
        super(CombinedNetwork_multi, self).__init__()

        self.normorder = normorder
        
        self.subnet1 = SubNetwork_bandpower_multi(input_sizes[0], subnet_hidden_sizes[0], dropout_rate=dropout_rates[0], normorder=normorder)
        self.subnet2 = SubNetwork_timeseries_multi(input_sizes[1], subnet_hidden_sizes[1], dropout_rate=dropout_rates[1])
        self.subnet3 = SubNetwork_categorical_multi(input_sizes[2], subnet_hidden_sizes[2], dropout_rate=dropout_rates[2])
        
        # Get the correct input size from variable hidden layers
        combined_subnet_size = subnet_hidden_sizes[0][-1]+subnet_hidden_sizes[1][-1]+subnet_hidden_sizes[2][-1]
        combined_hidden_size = int(combined_hidden_size*combined_subnet_size)

        # Final layers after combining the sub-networks
        self.fc_combined = nn.Linear(combined_subnet_size, combined_hidden_size)
        self.bn          = nn.BatchNorm1d(combined_hidden_size)
        self.relu        = nn.ReLU()
        self.tanh        = nn.Tanh()
        self.fc_output   = nn.Linear(combined_hidden_size, output_size)
        self.dropout     = nn.Dropout(p=0.4)
        self.softmax     = nn.Softmax(dim=1)
        self.sigmoid     = nn.Sigmoid()

        # Handle selection of activation layer
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
