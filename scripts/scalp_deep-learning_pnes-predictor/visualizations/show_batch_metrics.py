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

        print(DF)
        DF.to_csv("nn_metrics.csv",index=False)
    else:
        DF = PD.read_csv(argv[2])

        sns.boxplot(data=DF,x='folder',y='Test_Sensitivity')
        PLT.show()