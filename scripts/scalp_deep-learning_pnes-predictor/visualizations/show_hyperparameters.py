import json
import glob
import argparse
import pandas as PD
import pylab as PLT
import seaborn as sns
from sklearn.decomposition import PCA

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--result_dir", type=str, required=True, help="Folder path to the results.")
    args = parser.parse_args()

    # Read in the data
    files = glob.glob(f"{args.result_dir}*/result.json")
    DF    = PD.DataFrame()
    aucs  = []
    for idx,ifile in enumerate(files):
        try:
            jdata = json.load(open(ifile,'r'))
            iDF   = PD.DataFrame(jdata['config'],index=[idx])
            DF    = PD.concat((DF,iDF))
            aucs.append(jdata['Train_AUC'])
        except:
            pass

    # Quick mapping of strings
    norm_map = {'before':0,'after':1}
    act_map  = {'tanh':0,'relu':1}
    DF['normorder']  = DF['normorder'].apply(lambda x:norm_map[x])
    DF['activation'] = DF['activation'].apply(lambda x:act_map[x])

    # Get a projection to analyze
    pca              = PCA(n_components=2)
    new_DF           = PD.DataFrame(pca.fit_transform(DF),columns=['pca_0','pca_1'])
    new_DF['auc']    = aucs
    new_DF['cutoff'] = new_DF['auc'] > .6
    
    # Plot the results
    sns.scatterplot(data=new_DF,x='pca_0',y='pca_1',hue='cutoff')
    PLT.show()