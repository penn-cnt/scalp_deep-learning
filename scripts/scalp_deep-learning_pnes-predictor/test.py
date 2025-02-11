import pickle
import numpy as np
import pandas as PD
from sys import exit
from tqdm import tqdm

class test_class:

    def __init__(self,DF,model_blocks):

        self.DF           = DF
        self.time_horizon = 60
        self.t_step       = int(self.time_horizon/15)
        self.window_size  = 2*self.time_horizon
        self.time_col     = 't_start'
        self.feature_cols = model_blocks['frequency']+model_blocks['time']+model_blocks['categorical']
        self.target_cols  = model_blocks['target']
        self.ref_cols     = np.setdiff1d(DF.columns,self.feature_cols)
        self.ref_cols     = np.setdiff1d(self.ref_cols,self.target_cols)
        self.ref_cols     = list(np.setdiff1d(self.ref_cols,[self.time_col]))

        # Try to downcast the columns slightly for memory improvement and speed
        for icol in self.DF.columns:
            if self.DF[icol].dtype == 'int64':
                self.DF[icol] = PD.to_numeric(self.DF[icol],downcast='integer')
            if self.DF[icol].dtype == 'float64':
                self.DF[icol] = PD.to_numeric(self.DF[icol],downcast='float')

    def sample(self):

        # List to store transformed rows
        rows = []

        # Process each file group separately
        for _, group in tqdm(self.DF.groupby(self.ref_cols)):

            # Get the start times for this data slice
            group    = group.sort_values(self.time_col)  # Ensure sorting by time
            t_starts = group[self.time_col].to_numpy()  # Extract t_start as NumPy array

            for i, row in group.iterrows():
                t_start = row[self.time_col]

                # Use the t_start array to define the time window range
                time_diff   = np.abs(t_starts - t_start)
                window_mask = time_diff <= self.time_horizon

                # Get the window using iloc for efficient slicing
                window = group.iloc[window_mask]

                # If the window does not have enough neighbors, skip this row
                if (window['t_start'].max()-window['t_start'].min()) < self.window_size:
                    continue

                # Flatten features in the windowed range
                new_row = [row[icol] for icol in self.ref_cols] + [row[self.time_col]] + window[self.feature_cols].values.flatten().tolist() + [row[icol] for icol in self.target_cols]
                rows.append(new_row)

        # Generate new column names
        columns  = self.ref_cols+[self.time_col]
        columns += [f"{feat}_t{t_shift}" for t_shift in range(-self.t_step, self.t_step + 1) for feat in self.feature_cols]
        columns += self.target_cols

        # Create DataFrame
        result_df = PD.DataFrame(rows, columns=columns)

if __name__ == '__main__':

    outpath                   = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/MODELS/SSL/DATA/vector_data.pickle'
    model_blocks, train, test = pickle.load(open(outpath,'rb'))

    TC = test_class(train,model_blocks)
    TC.sample()