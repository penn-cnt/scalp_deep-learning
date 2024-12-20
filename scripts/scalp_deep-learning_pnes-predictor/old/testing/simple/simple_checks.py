import os
import nltk
import argparse
import numpy as np
import pandas as PD
import pylab as PLT
from tqdm import tqdm
from nltk.corpus import stopwords
from prettytable import PrettyTable 
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif,SelectKBest
from sklearn.linear_model import LogisticRegression,SGDClassifier
  
class simple_models:

    def __init__(self):
        pass

    def fit_report(self,clf):
        
        # Fit the data
        clf.fit(self.X_train_scaled, self.Y_train.flatten())
        
        # Predict the test data
        self.Y_test_pred = clf.predict(self.X_test_scaled)
        
        # Calculate the AUC
        auc = roc_auc_score(self.Y_test,self.Y_test_pred)
        print(f"AUC Score: {auc:.2f}")

    def simple_LR(self):
        
        # Alert user
        print("Performing a simple LR model.")
        
        # Make the fitter
        clf = LogisticRegression(solver='sag', max_iter=1000, random_state=0)
        
        # Fit and report
        self.fit_report(clf)

    def simple_sgd(self):

        # Alert user
        print("Performing a simple SGD model.")

        # Change the binary output label to work for hinge
        self.Y_train[self.Y_train==0] = -1
        self.Y_test[self.Y_test==0]   = -1

        # Make the fitter
        clf = SGDClassifier(random_state=0, max_iter=1000)

        # Fit and report
        self.fit_report(clf)

    def simple_anova(self):

        # Make the ANOVA dataframe
        f_statistic, p_values = f_classif(self.X_train[:,:-1], self.Y_train.flatten())
        rows                  = []
        for idx in range(f_statistic.size):
            rows.append([self.Xcols[idx],f_statistic[idx],p_values[idx]])
        anova_df = PD.DataFrame(rows,columns=['feature','f-stat','p-val'])
        
        # Sort the dataframe
        anova_df['f-stat'] = 1/anova_df['f-stat']
        anova_df           = anova_df.sort_values(by=['p-val','f-stat'])
        anova_df['f-stat'] = 1/anova_df['f-stat']
        anova_orig         = anova_df.copy()

        # Clean up the features for easier visualization
        anova_df['channel'] = anova_df['feature'].apply(lambda x:x.split('_')[0])
        anova_df['feature'] = anova_df['feature'].apply(lambda x:'_'.join(x.split('_')[2:]))

        # Show the user the results
        myTable = PrettyTable(["Feature",'Channel', "F-score", "P-val"]) 
        for idx in range(anova_df.shape[0]):
            irow   = anova_df.iloc[idx]
            newrow = [irow.feature,irow.channel,f"{irow['f-stat']:.2f}",f"{irow['p-val']:.2e}"]
            myTable.add_row(newrow)

        if self.verbose: print(myTable)
        return anova_orig

class simple_models_prep(simple_models):

    def __init__(self,indata,verbose):
        self.indata  = indata
        self.verbose = verbose

    def data_prep(self):
        self.map_targets()
        self.get_XY()
        self.split_data()
        self.transform_data()

    def simple_model_handler(self,mtype):

        if mtype == 'LR':
            simple_models.simple_LR(self)
        elif mtype == 'SGD':
            simple_models.simple_sgd(self)
        elif mtype == 'ANOVA':
            return simple_models.simple_anova(self)

    def transform_data(self,scaletype='SS'):
        
        if scaletype=='SS':
            SS = StandardScaler()
            self.X_train_scaled = SS.fit_transform(self.X_train)
            self.X_test_scaled  = SS.transform(self.X_test)
            
    def split_data(self):
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.raw_X, self.raw_Y, test_size=0.33, random_state=42)

    def get_XY(self):
        blacklist  = ['file', 't_start', 't_end', 't_window', 'uid', 'target']
        self.Xcols = np.setdiff1d(self.indata.columns,blacklist)
        self.Ycols = ['target']
        self.raw_X = self.indata[self.Xcols].values
        self.raw_Y = self.indata[self.Ycols].values

    def map_targets(self):
        self.tmap             = {'pnes':0,'epilepsy':1}
        self.indata['target'] = self.indata['target'].apply(lambda x:self.tmap[x])

    def return_data(self):
        return self.raw_X,self.X,self.raw_Y