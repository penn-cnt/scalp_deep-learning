import os
import pickle
import numpy as np
import pandas as PD
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

class data_manager:

    def __init__(self,inpath,pivotpath,mlppath,splitmethod):
        
        self.inpath      = inpath
        self.pivotpath   = pivotpath
        self.mlppath     = mlppath
        self.splitmethod = splitmethod

    def pivotdata_prep(self,normflag=False):
        """
        Workflow for creating a pivoted dataset.

        Args:
            normflag (bool, optional): Normalize spectral energy measurements. Defaults to False.

        Returns:
            pandas dataframe: A pivoted and cleaned up dataset with all the features for a given clip as columns.
        """

        if not os.path.exists(self.pivotpath):

            print("Creating pivot data.")            
            # Create the pivot dataset
            self.DF = PD.read_csv(self.inpath)
            self.get_sleep_state()
            self.drop_extra_raw_labels()
            self.drop_failed_runs()
            self.marsh_filter()
            self.drop_fullfile()
            self.define_columns()
            self.pivot()
            if normflag:self.normalized_bandpower()
            self.get_alpha_delta()
            self.shift_ref()
            self.DF.dropna(inplace=True)

            # Save the pivot dataset
            self.DF.to_pickle(self.pivotpath)
        else:
            self.DF = PD.read_pickle(self.pivotpath)
        return self.DF

    def mlpdata_prep(self):
        """
        Workflow for creating MLP inputs

        Returns:
            tuple: A tuple with the data, column headers, etc. for use in a DL network.
        """

        # Create the neural network inputs
        if not os.path.exists(self.mlppath):
            
            print("Creating MLP data.")

            # Create the MLP input
            self.map_targets()
            self.drop_extra_cols()
            self.define_column_groups()
            self.make_model_groups()
            self.data_rejection()
            self.apply_column_transform()
            self.sleep_to_categorical()
            self.target_to_categorical()
            self.define_inputs()

            # Create the output object
            out_object = (self.X_train_bandpower,self.X_test_bandpower,self.X_train_timeseries,self.X_test_timeseries,
                          self.X_train_categorical,self.X_test_categorical,self.Y_train,self.Y_test,self.model_block,
                          self.train_localization,self.test_localization)

            # Save the MLP input
            pickle.dump(out_object,open(self.mlppath,'wb'))
        else:
            out_object = pickle.load(open(self.mlppath,'rb'))
        return out_object

    ################################
    ##### Data Pivot functions #####
    ################################

    def normalized_bandpower(self):
        """
        Normalize the bandpowers using the sum across all bandpower features, excluding features that are already normalized. 
        """

        # Make a summed column for all the bandpowers in a row for a given channel
        print("Normalizing Bandpower")
        for ichannel in self.channels:
            col_mask = []
            for icol in self.DF.columns:
                if ichannel in icol and 'welch' in icol and 'normalized' not in icol:
                    col_mask.append(icol)
        
            # get the summed bandpower
            self.DF[f"{ichannel}_total_BP"] = self.DF[col_mask].values.sum(axis=1)

            # Remove nans and zeros
            self.DF.dropna(inplace=True)
            self.DF = self.DF.loc[self.DF[f"{ichannel}_total_BP"] > 0]

            # Create the normalized power
            for icol in col_mask:
                self.DF.loc[:,[icol]] = self.DF[icol].values/self.DF[f"{ichannel}_total_BP"].values
        
    def get_sleep_state(self):

        # Set up the word tokenizer
        stop_words      = set(stopwords.words('english'))
        tokenizer       = RegexpTokenizer(r'\w+')

        # Grab the annotation values
        annots      = self.DF['annotation'].values
        sleep_state = []
        for istr in annots:

            # Tokenize the annotations
            tokens          = tokenizer.tokenize(istr.lower())
            filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 1]

            # See if we can predict the sleep state from tokens
            sleep_pred = []
            for itoken in filtered_tokens:
                if itoken in ['sleep','asleep','n1','n2','n3','sws','rem','spindles']:
                    sleep_pred.append('sleep')
                if itoken in ['wake','eye','blink','pdr','eat','chew']:
                    sleep_pred.append('wake')
            
            # Add in the values to the output list
            if len(sleep_pred) >= 2:
                sleep_state.append(-1)
            elif len(sleep_pred) == 0:
                sleep_state.append(0)
            else:
                if sleep_pred[0] == 'sleep':
                    sleep_state.append(1)
                elif sleep_pred[0] == 'wake':
                    sleep_state.append(2)
        self.DF['sleep_state'] = sleep_state
        self.DF = self.DF.loc[self.DF.sleep_state!=-1]

    def drop_extra_raw_labels(self):

        self.DF.drop(['annotation','ieeg_file','ieeg_start_sec','ieeg_duration_sec'],axis=1,inplace=True)

    def drop_failed_runs(self):
        self.DF = self.DF.loc[self.DF.tag!='None']

    def marsh_filter(self):

        self.DF = self.DF.loc[self.DF.marsh_rejection==True].drop(['marsh_rejection'],axis=1)

    def drop_fullfile(self):

        self.DF = self.DF.loc[self.DF.t_window==30]

    def define_columns(self):
        blacklist       = ['file', 't_start', 't_end', 't_window', 'method', 'tag', 'uid', 'target','sleep_state']
        self.channels   = np.setdiff1d(self.DF.columns,blacklist)
        self.ref_cols   = ['file', 't_start', 't_end', 't_window', 'uid', 'target','sleep_state']
        self.pivot_cols = ['method', 'tag'] 

    def pivot(self):

        pivot_df         = self.DF.pivot_table(index=self.ref_cols,columns=self.pivot_cols,values=self.channels,aggfunc='first')
        pivot_df.columns = [f'{val}_{method}_{tag}' for val, method, tag in pivot_df.columns]
        self.DF          = pivot_df.reset_index()

    def get_alpha_delta(self):

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

    def return_data(self):
        return self.DF

    ##############################
    ##### MLP Prep functions #####
    ##############################

    def map_targets(self):
        self.tmap         = {'pnes':0,'epilepsy':1}
        self.DF['target'] = self.DF['target'].apply(lambda x:self.tmap[x])

    def drop_extra_cols(self):

        # Define drop columns
        drop_cols = ['t_end','t_start','t_window']
        self.DF = self.DF.drop(drop_cols,axis=1)

        # Drop raw bandpowers
        blacklist = []
        for icol in self.DF.columns:
            if 'spectral_energy_welch' in icol and 'normalized' not in icol:
                blacklist.append(icol)
        self.DF = self.DF.drop(blacklist,axis=1)

    def define_column_groups(self):
        """
        Define the modeling blocks based on column name.
        """

        # Store the columns into the right modeling block for preprocessing and initial fits.
        self.iso_cols        = []
        self.transform_block = {'standard_scaler_wlog10':[],'yeo-johnson':[],'passthrough':[]}
        self.model_block     = {'bandpower':[],'timeseries':[],'categoricals':[],'targets':[]}
        for icol in self.DF.columns:
            if 'normalized_spectral_energy_welch' in icol:
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
            else:
                self.transform_block['passthrough'].append(icol)

    def make_model_groups(self):

        if self.splitmethod == 'raw':
            # Make an index object for splitting
            DF_inds = np.arange(self.DF.shape[0])

            #self.train_raw, self.test_raw = train_test_split(self.DF, test_size=0.33, random_state=42)
            train_inds, test_inds = train_test_split(DF_inds, test_size=0.33, random_state=42)
        elif self.splitmethod == 'uid':
            # Split on uid
            splitter              = GroupShuffleSplit(test_size=.33, n_splits=1, random_state = 42)
            split                 = splitter.split(self.DF, groups=self.DF['uid'])
            train_inds, test_inds = next(split)

        # get the train and test indices
        self.train_raw = self.DF.iloc[train_inds]
        self.test_raw  = self.DF.iloc[test_inds]

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

        # Fix some typing issues from including the file
        self.train_transformed['sleep_state'] = self.train_transformed['sleep_state'].astype('int')
        self.test_transformed['sleep_state']  = self.test_transformed['sleep_state'].astype('int')
        self.train_transformed['target']      = self.train_transformed['target'].astype('int')
        self.test_transformed['target']       = self.test_transformed['target'].astype('int')


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
        self.train_localization  = self.train_transformed[['file','uid']]
        self.test_localization   = self.test_transformed[['file','uid']]