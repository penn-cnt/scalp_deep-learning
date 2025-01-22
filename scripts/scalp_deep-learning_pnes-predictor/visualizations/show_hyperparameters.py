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
    parser.add_argument("--outfile", type=str, required=True, help="Path for output dataframe csv.")
    parser.add_argument("--plotdir", type=str, required=True, help="Path for output plots.")
    parser.add_argument("--testcol", type=str, default='train_auc_clip', help="Path for output plots.")
    parser.add_argument("--cutoff", type=float, default=0.85, help="Path for output plots.")
    args = parser.parse_args()

    # Read in the data
    files             = glob.glob(f"{args.result_dir}*/result.json")
    DF                = PD.DataFrame()
    patient_auc_train = []
    clip_auc_train    = []
    patient_auc_test  = []
    clip_auc_test     = []    
    fpath             = []
    for idx,ifile in enumerate(files):
        try:
            jdata = json.load(open(ifile,'r'))
            iDF   = PD.DataFrame(jdata['config'],index=[idx])
            DF    = PD.concat((DF,iDF))
            patient_auc_train.append(jdata['Train_AUC'])
            clip_auc_train.append(jdata['Train_AUC_clip'])
            patient_auc_test.append(jdata['Test_AUC'])
            clip_auc_test.append(jdata['Test_AUC_clip'])
            fpath.append(ifile)
        except:
            pass

    # Quick mapping of strings
    norm_map   = {'before':0,'after':1}
    act_map    = {'tanh':0,'relu':1}
    thresh_map = {'posterior':0,'quantile':1}
    DF['normorder']  = DF['normorder'].apply(lambda x:norm_map[x])
    DF['activation'] = DF['activation'].apply(lambda x:act_map[x])
    DF['consensus_theshold_method'] = DF['consensus_theshold_method'].apply(lambda x:thresh_map[x])

    # Add in auc to bigger dataframe
    DF['fpath']    = fpath
    incols         = DF.columns
    outcols        = ['train_auc','train_auc_clip','test_auc','test_auc_clip']
    DF[outcols[0]] = patient_auc_train
    DF[outcols[1]] = clip_auc_train
    DF[outcols[2]] = patient_auc_test
    DF[outcols[3]] = clip_auc_test
    outcols.extend(incols)

    # Save the results
    DF = DF[outcols]
    DF = DF.sort_values(by=[args.testcol],ascending=False)
    DF.to_csv(args.outfile,index=False)

    # Make a slice of the dataframe where we show only good aucs
    cutoff  = 0.85
    good_DF = DF.loc[DF[args.testcol]>args.cutoff]

    # Make new diagnostic plots
    blacklist = ['train_auc','train_auc_clip','test_auc','test_auc_clip','fpath']
    plotcols  = np.setdiff1d(DF.columns,blacklist)

    # Plot the distributions
    for icol in plotcols:
        fig     = PLT.figure(dpi=100,figsize=(8.,6.))
        ax      = fig.add_subplot(111)
        allplot = sns.kdeplot(data=DF,x=icol,color='k',ax=ax)
        aucplot = sns.kdeplot(data=good_DF,x=icol,color='r',ax=ax)
        ax.legend(['All Parameters',f"{args.testcol}>{args.cutoff:0.2f}"])
        ax.set_xlabel(f"x={icol}")
        ax.set_ylabel(f"P(x)")
        ax.set_title(f"{icol} PDF")
        if icol == 'weight':
            ax.set_yscale('log')
            ax.set_xscale('log')
        PLT.savefig(f"{args.plotdir}/{icol}.png")
        PLT.close("all")