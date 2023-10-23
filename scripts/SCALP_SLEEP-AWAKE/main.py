import argparse
import numpy as np
import pandas as PD
import matplotlib.pyplot as PLT

class data_loader:

    def __init__(self,infile):

        self.data = PD.read_pickle(infile)
    
    def get_channels(self):

        black_list    = ['file','t_start','t_end','dt','method','tag','uid','target','annotation']
        self.channels = []
        for icol in self.data.columns:
            if icol not in black_list:
                self.channels.append(icol)
        return self.channels

    def get_state(self):

        annots = self.data.annotation.values
        sleep  = np.zeros(annots.size)
        awake  = sleep.copy()
        for idx,iannot in enumerate(annots):
            if iannot != None:
                ann = iannot.lower()
                if 'wake' in ann or 'awake' in ann or 'pdr' in ann:
                    awake[idx]=1
                if 'sleep' in ann or 'spindle' in ann or 'k complex' in ann or 'sws' in ann:
                    sleep[idx]=1
        self.data['sleep'] = sleep
        self.data['awake'] = awake

    def state_split(self):

        self.sleep_data = self.data.loc[(self.data.sleep==1)]
        self.awake_data = self.data.loc[(self.data.awake==1)]

    def recast(self):
        
        for icol in self.data.columns:
            try:
                self.data[icol]=self.data[icol].astype('float')
            except:
                pass

    def grouped_data(self):

        self.sleep_group = self.sleep_data.groupby(['uid','tag'],as_index=False)[self.channels].median()
        self.awake_group = self.awake_data.groupby(['uid','tag'],as_index=False)[self.channels].median()

        return self.sleep_group,self.awake_group

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--infile", help="Path to input file.")
    parser.add_argument("--outdir", help="Path to output directory.")
    args = parser.parse_args()

    DL       = data_loader(args.infile)
    channels = DL.get_channels()
    DL.get_state()
    DL.recast()
    DL.state_split()
    DF_sleep,DF_awake = DL.grouped_data()

    # Define the alpha delta band tags
    alpha='[8.0,12.0]'
    delta='[1.0,4.0]'

    # Get the alpha and delta for awake and sleep
    alpha_awake = DF_awake.loc[DF_awake.tag==alpha]
    delta_awake = DF_awake.loc[DF_awake.tag==delta]
    alpha_sleep = DF_sleep.loc[DF_sleep.tag==alpha]
    delta_sleep = DF_sleep.loc[DF_sleep.tag==delta]

    ratio_awake = []
    ratio_sleep = []
    for ichannel in channels:

        ratio_awake.append(alpha_awake[ichannel].values/delta_awake[ichannel].values)
        ratio_sleep.append(alpha_sleep[ichannel].values/delta_sleep[ichannel].values)

        fig = PLT.figure(dpi=100,figsize=(6.,6.))
        ax  = fig.add_subplot(111)
        ax.scatter(alpha_awake[ichannel].values,delta_awake[ichannel].values,color='g')
        ax.scatter(alpha_sleep[ichannel].values,delta_sleep[ichannel].values,color='r')
        ax.set_xlabel("Alpha (8-12 Hz)", fontsize=14)
        ax.set_ylabel("Delta (1-4 Hz)", fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(ichannel, fontsize=14)
        fig.tight_layout()
        PLT.savefig(args.outdir+"%s.png" %(ichannel))
        PLT.close("all")

    print("Wake",np.mean(ratio_awake),np.std(ratio_awake))
    print("Sleep",np.mean(ratio_sleep),np.std(ratio_sleep))