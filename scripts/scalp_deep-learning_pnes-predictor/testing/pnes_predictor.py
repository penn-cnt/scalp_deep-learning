import os
import nltk
import argparse
import numpy as np
import pandas as PD
import pylab as PLT
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier

# Local imports
from simple.simple_checks import *
from mlp.mlp_model import *

class data_manager:

    def __init__(self,inpath):
        
        self.rawdata = PD.read_pickle(inpath)

    def data_prep(self):

        self.DF = self.rawdata.copy()
        self.get_sleep_state()
        self.drop_extra_raw_labels()
        self.drop_failed_runs()
        self.marsh_filter()
        self.drop_fullfile()
        self.define_columns()
        self.pivot()
        self.get_alpha_delta()
        self.shift_ref()
        self.DF.dropna(inplace=True)

    def drop_failed_runs(self):
        self.DF = self.DF.loc[self.DF.tag!='None']
    
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

    def drop_fullfile(self):

        self.DF = self.DF.loc[self.DF.t_window==30]

    def drop_extra_raw_labels(self):

        self.DF.drop(['annotation','ieeg_file','ieeg_start_sec','ieeg_duration_sec'],axis=1,inplace=True)

    def define_columns(self):
        blacklist       = ['file', 't_start', 't_end', 't_window', 'method', 'tag', 'uid', 'target','sleep_state']
        self.channels   = np.setdiff1d(self.DF.columns,blacklist)
        self.ref_cols   = ['file', 't_start', 't_end', 't_window', 'uid', 'target','sleep_state']
        self.pivot_cols = ['method', 'tag'] 

    def marsh_filter(self):

        self.DF = self.DF.loc[self.DF.marsh_rejection==True].drop(['marsh_rejection'],axis=1)

    def pivot(self):

        pivot_df         = self.DF.pivot_table(index=self.ref_cols,columns=self.pivot_cols,values=self.channels,aggfunc='first')
        pivot_df.columns = [f'{val}_{method}_{tag}' for val, method, tag in pivot_df.columns]
        self.DF          = pivot_df.reset_index()

def quick_checks(DF,verbose=True):

    # Send the data to the model manager
    print("Using basic models with all columns.")
    SMP = simple_models_prep(DF,verbose=verbose)
    SMP.data_prep()

    # Basic models
    #SMP.simple_model_handler('LR')
    #SMP.simple_model_handler('SGD')

    # Get ANOVA columns
    print("Using ANOVA to get best features.")
    anova_df = SMP.simple_model_handler('ANOVA')
    return anova_df

if __name__ == '__main__':
    
    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--feature_file", type=str, required=True, help="Filepath to the feature file.")
    parser.add_argument("--ifile", type=str, help="Intermediate file to skip cleanup.")
    args = parser.parse_args()

    # Prepare the data
    if args.ifile == None or not os.path.exists(args.ifile):
        print("Formatting the data for use in models.")
        DM = data_manager(args.feature_file)
        DM.data_prep()
        DF = DM.return_data()
        if args.ifile != None:
            DF.to_pickle(args.ifile)
    else:
        print("Reading in existing model inputs.")
        DF = PD.read_pickle(args.ifile)

    # Perform some simple models to rule them out
    #anova_df = quick_checks(DF.copy(),verbose=False)
    
    # Remove the columns that anova flagged as non-informative
    #bad_cols = anova_df.loc[anova_df['p-val']>0.05]['feature'].values
    #DF       = DF.drop(bad_cols,axis=1)

    # Perform an MLP model
    MLPP = mlp_prep(DF)
    MLPP.processing_pipeline()
    MLPP.call_mlp()