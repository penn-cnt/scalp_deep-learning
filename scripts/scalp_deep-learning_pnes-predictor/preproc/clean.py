import os
import re
import ast
import yaml
import pickle
import numpy as np
import pandas as PD
import pylab as PLT
import seaborn as sns
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split,GroupShuffleSplit 
from sklearn.preprocessing import StandardScaler,LabelBinarizer,PowerTransformer

class pivot_manager:
    """
    Class for cleaning and pivoting the raw feature data to make columns for all features in one row for each clip (versus unique rows for each feature and clip).
    """

    def __init__(self,inpath,pivotpath,clip_length):
        """
        Initialize the pivot manager class with some relevant filepaths.

        Args:
            inpath (_type_): Path to the raw feature csv
            pivotpath (_type_): Path to save the pivoted data to.
        """
        
        self.inpath      = inpath
        self.pivotpath   = pivotpath
        self.clip_length = clip_length

    def workflow(self):
        """
        Workflow for creating a pivoted dataset.

        Returns:
            pandas dataframe: A pivoted and cleaned up dataset with all the features for a given clip as columns.
        """

        if not os.path.exists(self.pivotpath):

            print("Creating pivot data.")            
            # Create the pivot dataset
            self.DF = PD.read_csv(self.inpath)
            self.drop_extra_raw_labels()
            self.drop_failed_runs()
            self.keep_clips_only()
            self.define_pivot()
            self.pivot()
            self.get_alpha_delta()
            self.shift_ref()
            self.expand_targets()
            self.make_uid()
            self.drop_extra_pivot_labels()
            self.DF.dropna(inplace=True)
            
            # Save the pivot dataset
            self.DF.to_pickle(self.pivotpath)
        else:
            self.DF = PD.read_pickle(self.pivotpath)
        return self.DF

    def drop_extra_raw_labels(self):
        """
        Drop a few unnecessary columns from the raw feature dataframe.
        """

        self.DF.drop(['annotation','marsh_rejection'],axis=1,inplace=True)

    def drop_failed_runs(self):
        """
        Remove any clips with failed feature extractions.
        """

        self.DF = self.DF.loc[self.DF.tag!='None']

    def keep_clips_only(self):
        """
        Remove any clips with a different clip length that needed for the experiment. This can easily arise if using longer clips to get some baseline metrics (i.e. Marsh Filtering) which are in the raw feature dataframe.
        """

        self.DF = self.DF.loc[self.DF.t_window==self.clip_length]

    def define_pivot(self):
        """
        Define the columns needed to make a pivot dataframe.
        """

        # Columns with known identifiers (not channel features)
        blacklist       = ['file', 't_start', 't_end', 't_window', 'method', 'tag', 'target','yasa_prediction']

        # GHHet the channel column names
        self.channels   = np.setdiff1d(self.DF.columns,blacklist)

        # Define the reference columns and the pivot columns
        self.ref_cols   = ['file', 't_start', 't_end', 't_window', 'target','yasa_prediction']
        self.pivot_cols = ['method', 'tag'] 

    def pivot(self):
        """
        Perform the dataframe pivot and make new column labels that combine channel label and feature name
        """

        pivot_df         = self.DF.pivot_table(index=self.ref_cols,columns=self.pivot_cols,values=self.channels,aggfunc='first')
        pivot_df.columns = [f'{val}_{method}_{tag}' for val, method, tag in pivot_df.columns]
        self.DF          = pivot_df.reset_index()

    def get_alpha_delta(self):
        """
        Calculate the alpha delta ratio for each clip.
        """

        # Make a blank array for later averaging
        blank_ad = np.zeros((self.DF.shape[0],self.channels.size))

        # Loop over channels to get their respective alpha and delta
        for idx,ichannel in enumerate(self.channels):

            # Loop over columns to find the right pair
            for icol in self.DF.columns:
                if ichannel in icol:
                    if '[8.00,13.00]' in icol:
                        alpha_col = icol
                    if '[1.00,4.00]' in icol:
                        delta_col = icol
            blank_ad[:,idx] = self.DF[alpha_col].values/self.DF[delta_col].values

        # Store the stat to the model dataframe
        for idx,ichannel in enumerate(self.channels):
            self.DF[f"{ichannel}_AD_AD"] = blank_ad[:,idx]

    def shift_ref(self):
        """
        Shift the quantiles so they are all referenced to the median of their observation. Removes some per file deviations.
        """

        for ichannel in self.channels:
            for icol in self.DF.columns:
                if ichannel in icol and 'median' in icol:
                    med_ref = icol
                if ichannel in icol and 'quantile_0.25' in icol:
                    q25_ref = icol
                if ichannel in icol and 'quantile_0.75' in icol:
                    q75_ref = icol
            self.DF[q75_ref] = self.DF[q75_ref]-self.DF[med_ref]
            self.DF[q25_ref] = self.DF[q25_ref]-self.DF[med_ref]

    def expand_targets(self):
        """
        New data repository uses a dictionary to store a lot more target data. Need to make these into discrete columns.
        """

        # Sample the new target labels from the first entry, make objects to store results
        rawstr = self.DF.iloc[0].target

        # Clean up the string. Stored targets dont have well formed string denotation.
        target_cols = ast.literal_eval(rawstr.replace('{','{"').replace('}','"}').replace(':','":"').replace(',','","')).keys()

        # Make the object we will attach to the dataframe
        output = {}
        for icol in target_cols:
            output[f"target_{icol}"] = []

        # Make a dictionary to store the new target info
        for ii in range(self.DF.shape[0]):
            
            # Get the target and clean it up
            rawtarget   = self.DF.iloc[ii].target
            cleantarget = rawtarget.replace('{','{"').replace('}','"}').replace(':','":"').replace(',','","')

            # Add the results to the output object
            targets = ast.literal_eval(cleantarget)
            for icol in targets.keys():
                output[f"target_{icol}"].append(targets[icol])
            
        # Concatenate the results
        self.DF = PD.concat((self.DF,PD.DataFrame(output)),axis=1)

    def make_uid(self):
        """
        Make a unique identifier for each patient based on the filepath.
        """

        # Define the regular expression used to find the subject number
        pattern = r'HUP(\d+)_'

        # Loop over the filenames to get the new uid column
        self.DF['uid'] = self.DF['file'].apply(lambda x:int(re.search(pattern, x).group(1)))

    def drop_extra_pivot_labels(self):
        """
        Drop any extraneous columns not needed now that we have a pivoted datafrme.
        """

        self.DF.drop(['t_end', 't_window','target'],axis=1,inplace=True)

class vector_manager:
    """
    Class for making the actual input vectors to whatever DL model we choose to use.
    """

    def __init__(self, pivot_DF, target_col, vector_path, criteria_path, mapping_path, transformer_path, vector_plot_dir):

        # Save some class variables from the front-end
        self.DF               = pivot_DF
        self.target_col       = target_col
        self.vector_path      = vector_path
        self.criteria_path    = criteria_path
        self.mapping_path     = mapping_path
        self.transformer_path = transformer_path
        self.vector_plot_dir  = vector_plot_dir

        # Save some hard-coded class variables
        self.split_method     = 'uid'
        self.outlier_method   = 'iso'
        self.leading_cols     = ['file','t_start','uid']

    def workflow(self):
        """
        Workflow for creating MLP inputs

        Returns:
            tuple: A tuple with the data, column headers, etc. for use in a DL network.
        """

        # Create the neural network inputs
        if not os.path.exists(self.vector_path):
            
            print("Creating Vector data.")
            train_datasets = []
            test_datasets  = []

            # Create the MLP input up to the branching point for batch scoring
            self.apply_criteria()
            self.select_target()
            self.map_columns()
            self.define_column_groups()

            # 
            for batch_num in range(10):
                self.split_model_group(batch_num)
                self.outlier_rejection()
                self.apply_column_transform()
                self.encode_categoricals()

                # Plot the vectors as need
                if self.vector_plot_dir != None:
                    self.vector_plots()

                # Try to downcast the columns slightly for memory improvement and speed
                for icol in self.train_transformed.columns:
                    if self.train_transformed[icol].dtype == 'int64':
                        self.train_transformed[icol] = PD.to_numeric(self.train_transformed[icol],downcast='integer')
                        self.test_transformed[icol]  = PD.to_numeric(self.test_transformed[icol],downcast='integer')
                    if self.train_transformed[icol].dtype == 'float64':
                        self.train_transformed[icol] = PD.to_numeric(self.train_transformed[icol],downcast='float')
                        self.test_transformed[icol]  = PD.to_numeric(self.test_transformed[icol],downcast='float')

                # Append the batch dataset to the output list
                train_datasets.append(self.train_transformed)
                test_datasets.append(self.test_transformed)

            # Package the DL input object
            DL_object = (self.model_block,train_datasets,test_datasets)
            
            # Save the DL object
            pickle.dump(DL_object,open(self.vector_path,"wb"))
        else:
            DL_object = pickle.load(open(self.vector_path,"rb"))
        return DL_object

    def apply_criteria(self):
        """
        Apply any criteria for data usage in the model. This could be epilepsy diagnosis, type, lateralization, gender, etc.
        """

        if self.criteria_path != None:
            
            # Load the criteria dictionary
            criteria = yaml.safe_load(open(self.criteria_path,'r'))
            
            # Loop over the criteria columns and apply the criteria lists
            for icol in criteria.keys():
                self.DF = self.DF.loc[self.DF[icol].isin(criteria[icol])]

    def select_target(self):
        """
        Remove extra target columns and only use the singular target column the user wants.
        Note: This is after apply criteria so criteria can be used with other targets (i.e. Laterality, etc.)
        """

        # Find all the target columns
        target_cols = [icol for icol in self.DF.columns if 'target' in icol and icol != self.target_col]
        
        # Drop unused target cols
        self.DF.drop(target_cols,axis=1,inplace=True)

    def map_columns(self):

        # Load the mapping file
        mapping_config = yaml.safe_load(open(self.mapping_path,'r'))

        # Apply the mappings
        for icol in mapping_config.keys():
            self.DF[icol] = self.DF[icol].apply(lambda x:mapping_config[icol][x])

    def define_column_groups(self):
        """
        Define the modeling blocks based on column name.
        """

        # Load the transformer block configuration
        self.transformer_config = yaml.safe_load(open(self.transformer_path,'r'))
        
        # Define the transformer block names according to the configuration keys
        self.model_block = {key: [] for key in self.transformer_config.keys()}

        # Use the transformer config to find the relevant columns
        for icol in self.DF.columns:

            # Make a case for a catch-all passthrough option
            passflag = True
            
            # Step through the difference transformer options to assign the column name
            for itransformer in self.transformer_config.keys():
                for substr in self.transformer_config[itransformer]['cols']:
                    if substr in icol:
                        self.model_block[itransformer].append(icol)
                        passflag = False

            # If the column has not been handled by the transformer config, make it a passthrough
            if passflag:
                self.model_block['passthrough'].append(icol)
    
    def split_model_group(self,batch_num):
        """
        Split the model group randomly, by patient, or by time.
        """

        if self.split_method == 'random':
            # Make an index object for splitting
            DF_inds = np.arange(self.DF.shape[0])

            #self.train_raw, self.test_raw = train_test_split(self.DF, test_size=0.33, random_state=42)
            train_inds, test_inds = train_test_split(DF_inds, test_size=0.33, random_state=42+batch_num)
        elif self.split_method == 'uid':
            # Split on uid
            splitter              = GroupShuffleSplit(test_size=.33, n_splits=1, random_state = 42+batch_num)
            split                 = splitter.split(self.DF, groups=self.DF['uid'])
            train_inds, test_inds = next(split)

        # get the train and test indices
        self.train_raw = self.DF.iloc[train_inds]
        self.test_raw  = self.DF.iloc[test_inds]

        print(self.train_raw.shape)
        print("====")

    def outlier_rejection(self,contamination_factor=0.05):
        """
        Apply outlier rejection to the dataset
        """

        if self.outlier_method == 'iso':

            # Alert user
            print("Running isolation forest to reject the most extreme time segments.")

            # Get the columns we can use for outlier rejection
            iso_cols = []
            for itransformer in self.transformer_config.keys():
                if self.transformer_config[itransformer]['iso']:
                    iso_cols.extend(self.model_block[itransformer])

            # Create and fit the isolation forest
            self.ISO = IsolationForest(contamination=contamination_factor, random_state=42).fit(self.train_raw[iso_cols].values)

            # Get the training mask
            train_mask     = (self.ISO.predict(self.train_raw[iso_cols].values)==1)
            self.train_raw = self.train_raw.iloc[train_mask]
            
            # Get the testing mask
            test_mask     = (self.ISO.predict(self.test_raw[iso_cols].values)==1)
            self.test_raw = self.test_raw.iloc[test_mask]

            print(f"Isolation Forest reduced training size to {train_mask.sum()} from {train_mask.size} samples.")
            print(f"Isolation Forest reduced test size to {test_mask.sum()} from {test_mask.size} samples.")

    def apply_column_transform(self):
        """
        Apply the scaling transformations to the transformer blocks.
        """

        # Create the column transformer list
        transformer_list = []

        # Loop over the model blocks to get the right transform type
        for itransformer in self.transformer_config.keys():
            method = self.transformer_config[itransformer]['method']

            # Case statements for different transform types
            if method == 'standard_scaler_wlog10':
                transformer_list.append((f"{itransformer}_{method}", StandardScaler(), self.model_block[itransformer]))
            elif method == 'yeo-johnson':
                transformer_list.append((f"{itransformer}_{method}", PowerTransformer('yeo-johnson'), self.model_block[itransformer]))
            else:
                transformer_list.append((f"{itransformer}_{method}", 'passthrough', self.model_block[itransformer]))

        # Create the column transformation actions
        ct = ColumnTransformer(transformer_list)

        # Apply the log transforms if requested
        for itransformer in self.transformer_config.keys():
            if self.transformer_config[itransformer]['log10']:
                for icol in self.model_block[itransformer]:
                    self.train_raw[icol] = np.log10(self.train_raw[icol].values)
                    self.test_raw[icol]  = np.log10(self.test_raw[icol].values)

        # Convert the data
        print("Applying distribution scaling.")
        ct.fit(self.train_raw)
        train_transformed = ct.transform(self.train_raw)
        test_transformed  = ct.transform(self.test_raw)

        # Make a new flat column header
        flat_cols = [x for xs in ct._columns for x in xs]
        self.train_transformed = PD.DataFrame(train_transformed,columns=flat_cols)
        self.test_transformed  = PD.DataFrame(test_transformed,columns=flat_cols)

        # Make a more easily parsed structure
        trailing_cols = np.setdiff1d(self.train_transformed.columns,self.leading_cols)
        cols          = self.leading_cols.copy()
        cols.extend(trailing_cols)

        # Reorder the columns
        self.train_transformed = self.train_transformed[cols]
        self.test_transformed  = self.test_transformed[cols]

        # Fix typings
        for icol in self.train_transformed.columns:
            try:
                self.train_transformed[icol] = PD.to_numeric(self.train_transformed[icol])
                self.test_transformed[icol]  = PD.to_numeric(self.test_transformed[icol])
            except:
                pass

    def encode_categoricals(self):
        """
        Convert categoricals with an encoder.
        """

        # Find the transformer blocks that need encoding
        for itransformer in self.transformer_config.keys():
            if self.transformer_config[itransformer]['method'] == 'encoder':

                # Make the new output model block object
                new_model_block = []

                # Loop over the columns in this model block
                for icol in self.model_block[itransformer]:

                    # Apply the label binarizer to the sleep labels
                    LB = LabelBinarizer()
                    LB.fit(self.train_transformed[icol])
                    encoder_labels = [f"{icol}_{int(x):02d}" for x in LB.classes_]
                    new_model_block.extend(encoder_labels)

                    # Add the vectors to train
                    encoded_data = LB.transform(self.train_transformed[icol])
                    if len(encoder_labels) == 2:
                        encoded_data = np.hstack((encoded_data, 1 - encoded_data))
                    train_vectors = PD.DataFrame(encoded_data,columns=encoder_labels)
                    self.train_transformed.drop([icol],axis=1,inplace=True)
                    self.train_transformed = PD.concat((self.train_transformed,train_vectors),axis=1)

                    # Add the vectors to test
                    encoded_data = LB.transform(self.test_transformed[icol])
                    if len(encoder_labels) == 2:
                        encoded_data = np.hstack((encoded_data, 1 - encoded_data))
                    test_vectors = PD.DataFrame(encoded_data,columns=encoder_labels)
                    self.test_transformed.drop([icol],axis=1,inplace=True)
                    self.test_transformed = PD.concat((self.test_transformed,test_vectors),axis=1)

                # Update the stored model block
                self.model_block[itransformer] = new_model_block

    def vector_plots(self):
        """
        Creates a series of plots of the input vectors to ensure well formed inputs.
        """
        
        # Make sure the plot directory exists first. If not, confirm with user if we should make it.
        if not os.path.isdir(self.vector_plot_dir):
            
            # Get input from the user as to what to do
            print(f"Vector output plot directory {self.vector_plot_dir} does not exist.")
            
            # Enter loop to get acceptable inputs
            while True:
                user_input = input(f"Create this directory (Yy/Nn)? ").lower()

                if user_input == 'y':
                    os.system(f"mkdir -p {self.vector_plot_dir}")
                    break
                elif user_input=='n':
                    return
                
        # Create the vector plots
        for itransformer in self.transformer_config.keys():
            if self.transformer_config[itransformer]['plot']:
                for icol in self.model_block[itransformer]:

                    # Make the title string
                    title_str  = f"Train: {icol}"
                    title_str += '\n'
                    title_str += f"{self.transformer_config[itransformer]['method']}"

                    # Make the plotting objects
                    fig = PLT.figure(dpi=100,figsize=(8.,6.))
                    ax  = fig.add_subplot(111)
                    sns.histplot(data=self.train_transformed,x=icol)
                    ax.set_title(title_str)
                    PLT.savefig(f"{self.vector_plot_dir}/train_{icol}.png")
                    PLT.close("all")
