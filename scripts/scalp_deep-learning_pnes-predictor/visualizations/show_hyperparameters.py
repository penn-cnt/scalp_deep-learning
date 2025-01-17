import json
import glob
import argparse
import numpy as np
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
    fpath = []
    for idx,ifile in enumerate(files):
        try:
            jdata = json.load(open(ifile,'r'))
            iDF   = PD.DataFrame(jdata['config'],index=[idx])
            DF    = PD.concat((DF,iDF))
            aucs.append(jdata['Train_AUC'])
            fpath.append(ifile)
        except:
            pass

    # Quick mapping of strings
    norm_map = {'before':0,'after':1}
    act_map  = {'tanh':0,'relu':1}
    DF['normorder']  = DF['normorder'].apply(lambda x:norm_map[x])
    DF['activation'] = DF['activation'].apply(lambda x:act_map[x])

    # Add in auc to bigger dataframe
    DF['fpath']    = fpath
    incols         = DF.columns
    outcols        = ['train_auc']
    DF[outcols[0]] = aucs
    outcols.extend(incols)

    # Save the results
    DF = DF[outcols]
    DF = DF.sort_values(by=['train_auc'],ascending=False)
    DF.to_csv("/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/MODELS/HYPERPARAMETERS/model_hyperparameters.csv",index=False)

    # Make a slice of the dataframe where we show only good aucs
    cutoff  = 0.85
    good_DF = DF.loc[DF.train_auc>cutoff]

    # Make new diagnostic plots
    blacklist = ['train_auc','fpath']
    plotcols  = np.setdiff1d(DF.columns,blacklist)

    # Plot the distributions
    outdir = '/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/MODELS/HYPERPARAMETERS/PLOTS'
    for icol in plotcols:
        fig     = PLT.figure(dpi=100,figsize=(8.,6.))
        ax      = fig.add_subplot(111)
        allplot = sns.kdeplot(data=DF,x=icol,color='k',ax=ax)
        aucplot = sns.kdeplot(data=good_DF,x=icol,color='r',ax=ax)
        ax.legend(['All Parameters',f"AUC>{cutoff:0.2f}"])
        ax.set_xlabel(f"x={icol}")
        ax.set_ylabel(f"P(x)")
        ax.set_title(f"{icol} PDF")
        if icol == 'weight':
            ax.set_yscale('log')
            ax.set_xscale('log')
        PLT.savefig(f"{outdir}/{icol}.png")
        PLT.close("all")
