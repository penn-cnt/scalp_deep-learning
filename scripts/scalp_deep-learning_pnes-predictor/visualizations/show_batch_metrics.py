import pandas as PD
import pylab as PLT
from sys import argv
import seaborn as sns
from pathlib import Path
from torch import load as tload

if __name__ == '__main__':

    if len(argv) == 2:
        # get the list of checkpoints
        files = []
        for path in Path(argv[1]).rglob('*.pth'):
            files.append(str(path))

        print(files)
        exit()

        # Make a dataframe
        DF             = PD.DataFrame()
        DF['folder']   = [ival.split('/train_pnes_handler_')[1].split('_')[0] for ival in files]
        DF['checknum'] = [ival.split('checkpoint_')[1].split('/')[0] for ival in files]

        # Get the metrics
        sensitivity = []
        npv         = []
        for ifile in files:
            metrics = tload(ifile)['metrics']
            sensitivity.append(metrics["Test_Sen"])
            npv.append(metrics["Test_NPV"])
        DF['Test_Sensitivity'] = sensitivity
        DF['Test_NPV']         = npv

        # Save the results
        print(f"{DF.shape[0]} Records")
        DF.to_csv("nn_metrics.csv",index=False)
    else:

        # Read in the data
        DF = PD.read_csv(argv[2])

        # Get the median sensitivity
        DF['median'] = DF.groupby('folder')['Test_Sensitivity'].transform('median')

        # Apply filtering as needed
        DF = DF.loc[(DF['median']>0.7)]
        #DF = DF.groupby("folder").filter(lambda group: group['Test_Sensitivity'].min() > 0.5)
        DF = DF.groupby("folder").filter(lambda group: group['Test_Sensitivity'].nsmallest(2).iloc[-1] > 0)


        # Make a combined metrics
        DF_long = DF.melt(id_vars=['folder', 'checknum','median'], value_vars=['Test_Sensitivity', 'Test_NPV'], 
                   var_name='metric', value_name='metric_value')

        # Plot results        
        fig = PLT.figure(dpi=100.,figsize=(12.,8.))
        ax1 = fig.add_subplot(111)
        sns.boxplot(data=DF_long,x='folder',y='metric_value',hue='metric',ax=ax1,whis=(2.5, 97.5))

        PLT.xticks(rotation='vertical')
        PLT.legend(loc='lower right')
        PLT.show()