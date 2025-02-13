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
        DF.to_csv("nn_metrics.csv",index=False)
    else:

        # Read in the data
        DF = PD.read_csv(argv[2])

        # Make a combined metrics
        DF_long = DF.melt(id_vars=['folder', 'checknum'], value_vars=['Test_Sensitivity', 'Test_NPV'], 
                   var_name='metric', value_name='metric_value')

        fig = PLT.figure(dpi=100.,figsize=(8.,6.))
        ax1 = fig.add_subplot(111)
        sns.boxplot(data=DF_long,x='folder',y='metric_value',hue='metric',ax=ax1)

        PLT.xticks(rotation='vertical')
        PLT.legend(loc='lower right')
        PLT.show()