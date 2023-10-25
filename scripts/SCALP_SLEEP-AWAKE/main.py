import os
import glob
import argparse
import numpy as np
import pandas as PD
from sys import exit
import matplotlib.pyplot as PLT
from scipy.signal import find_peaks
import matplotlib.colors as mcolors

class data_analysis:

    def __init__(self):

        pass

    def peaks(self, vals, prominence=1, width=3):

        return find_peaks(vals, prominence=prominence, width=width)

    def plot_raw_distributions(self):

        # First split according to the unique id
        uids = np.intersect1d(self.sleep_data['uid'].unique(),self.awake_data['uid'].unique())
        for i_uid in uids[:2]:

            # Get the awake and sleep splits
            awake_df = self.awake_data.loc[(self.awake_data.uid==i_uid)]
            sleep_df = self.sleep_data.loc[(self.sleep_data.uid==i_uid)]

            # Get the alpha and delta dataslices
            awake_alpha = awake_df.loc[(awake_df.tag==self.alpha_str)]
            awake_delta = awake_df.loc[(awake_df.tag==self.delta_str)]
            sleep_alpha = sleep_df.loc[(sleep_df.tag==self.alpha_str)]
            sleep_delta = sleep_df.loc[(sleep_df.tag==self.delta_str)]

            # Loop over the channels and make plots
            for ichannel in self.channels:
                awake_alpha_channel = awake_alpha[ichannel].values
                awake_delta_channel = awake_delta[ichannel].values
                sleep_alpha_channel = sleep_alpha[ichannel].values
                sleep_delta_channel = sleep_delta[ichannel].values

                # Make a better plotting baseline
                logbins = np.logspace(7,12,50)
                log_x   = (0.5*(logbins[1:]+logbins[:-1]))

                # Get the histogram counts
                awake_alpha_cnt = np.histogram(awake_alpha_channel,bins=logbins)[0]
                awake_delta_cnt = np.histogram(awake_delta_channel,bins=logbins)[0]
                sleep_alpha_cnt = np.histogram(sleep_alpha_channel,bins=logbins)[0]
                sleep_delta_cnt = np.histogram(sleep_delta_channel,bins=logbins)[0]
                
                # Get the peak information
                awake_alpha_peaks, awake_alpha_properties = self.peaks(awake_alpha_cnt)
                awake_delta_peaks, awake_delta_properties = self.peaks(awake_delta_cnt)
                sleep_alpha_peaks, sleep_alpha_properties = self.peaks(sleep_alpha_cnt)
                sleep_delta_peaks, sleep_delta_properties = self.peaks(sleep_delta_cnt)

                # Assign colors for easy pairing visually later
                awake_alpha_properties['color'] = []
                awake_delta_properties['color'] = []
                sleep_alpha_properties['color'] = []
                sleep_delta_properties['color'] = []
                color_cntr = 0
                unique_cnt = len(awake_alpha_peaks)+len(awake_delta_peaks)+len(sleep_alpha_peaks)+len(sleep_delta_peaks)
                while color_cntr < unique_cnt:
                    if color_cntr < len(awake_alpha_peaks):
                        awake_alpha_properties['color'] = self.color_names[color_cntr]
                    elif color_cntr < len(awake_alpha_peaks)+len(awake_delta_peaks):
                        awake_delta_properties['color'] = self.color_names[color_cntr]
                    elif color_cntr < len(awake_alpha_peaks)+len(awake_delta_peaks)+len(sleep_alpha_peaks):
                        sleep_alpha_properties['color'] = self.color_names[color_cntr]
                    else:
                        sleep_delta['color'] = self.color_names[color_cntr]
                print(awake_alpha_peaks)
                print(awake_alpha_properties)
                exit()

                # Make the plot canvas
                fig = PLT.figure(dpi=100,figsize=(10.,8.))
                ax1 = fig.add_subplot(221)
                ax2 = fig.add_subplot(222,sharex=ax1)
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224,sharex=ax3)

                # Histograms
                ax1.hist(awake_alpha_channel,bins=logbins,histtype='step',color='k',label="Awake Alpha")
                ax2.hist(awake_delta_channel,bins=logbins,histtype='step',color='k',label="Awake Delta")
                ax3.hist(sleep_alpha_channel,bins=logbins,histtype='step',color='k',label="Sleep Alpha")
                ax4.hist(sleep_delta_channel,bins=logbins,histtype='step',color='k',label="Sleep Delta")
                
                # Peak finder objects vertical
                ymin1 = awake_alpha_cnt[awake_alpha_peaks] - awake_alpha_properties["prominences"]
                ymin2 = awake_delta_cnt[awake_delta_peaks] - awake_delta_properties["prominences"]
                ymin3 = sleep_alpha_cnt[sleep_alpha_peaks] - sleep_alpha_properties["prominences"]
                ymin4 = sleep_delta_cnt[sleep_delta_peaks] - sleep_delta_properties["prominences"]
                ymax1 = awake_alpha_cnt[awake_alpha_peaks]
                ymax2 = awake_delta_cnt[awake_delta_peaks]
                ymax3 = sleep_alpha_cnt[sleep_alpha_peaks]
                ymax4 = sleep_delta_cnt[sleep_delta_peaks]
                ax1.vlines(x=log_x[awake_alpha_peaks], ymin=ymin1, ymax = ymax1, color = "r", lw=2)
                ax2.vlines(x=log_x[awake_delta_peaks], ymin=ymin2, ymax = ymax2, color = "r", lw=2)
                ax3.vlines(x=log_x[sleep_alpha_peaks], ymin=ymin3, ymax = ymax3, color = "r", lw=2)
                ax4.vlines(x=log_x[sleep_delta_peaks], ymin=ymin4, ymax = ymax4, color = "r", lw=2)

                # Peak finder objects horizontal
                xmin1 = [log_x[int(np.floor(ival))] for ival in awake_alpha_properties["left_ips"]]
                xmin2 = [log_x[int(np.floor(ival))] for ival in awake_delta_properties["left_ips"]]
                xmin3 = [log_x[int(np.floor(ival))] for ival in sleep_alpha_properties["left_ips"]]
                xmin4 = [log_x[int(np.floor(ival))] for ival in sleep_delta_properties["left_ips"]]
                xmax1 = [log_x[int(np.ceil(ival))] for ival in awake_alpha_properties["right_ips"]]
                xmax2 = [log_x[int(np.ceil(ival))] for ival in awake_delta_properties["right_ips"]]
                xmax3 = [log_x[int(np.ceil(ival))] for ival in sleep_alpha_properties["right_ips"]]
                xmax4 = [log_x[int(np.ceil(ival))] for ival in sleep_delta_properties["right_ips"]]
                ax1.hlines(y=awake_alpha_properties["width_heights"], xmin=xmin1, xmax=xmax1, color = "r")
                ax2.hlines(y=awake_delta_properties["width_heights"], xmin=xmin2, xmax=xmax2, color = "r")
                ax3.hlines(y=sleep_alpha_properties["width_heights"], xmin=xmin3, xmax=xmax3, color = "r")
                ax4.hlines(y=sleep_delta_properties["width_heights"], xmin=xmin4, xmax=xmax4, color = "r")

                # Final plot prep
                ax1.set_title("Patient id: %03d, Channel: %s" %(i_uid,ichannel), fontsize=13)
                ax1.legend(loc=1)
                ax2.legend(loc=1)
                ax3.legend(loc=1)
                ax4.legend(loc=1)
                ax1.set_xscale('log')
                ax2.set_xscale('log')
                ax3.set_xscale('log')
                ax4.set_xscale('log')
                fig.tight_layout()
                PLT.savefig(self.plotdir+"%03d_%s.png" %(i_uid,ichannel))
                PLT.close("all")

class data_loader(data_analysis):

    def __init__(self,infile, outdir):

        self.data      = PD.read_pickle(infile)
        self.outdir    = outdir
        self.alpha_str = '[8.0,12.0]'
        self.delta_str = '[1.0,4.0]'
        self.plotdir   = self.outdir+"PLOTS/"
        os.system("mkdir -p %s" %(self.plotdir))

        # Define the colorset
        self.color_names = np.array(list(mcolors.CSS4_COLORS.keys()))
        self.color_names = self.color_names[(self.color_names!='black')]
    
    def get_channels(self):
        """
        Determine channel names.

        Returns:
            List of channel names
        """

        black_list    = ['file','t_start','t_end','dt','method','tag','uid','target','annotation','sleep','awake']
        self.channels = []
        for icol in self.data.columns:
            if icol not in black_list:
                self.channels.append(icol)
        return self.channels

    def get_state(self):

        annots = self.data.annotation.values
        uannot = self.data.annotation.unique()
        sleep  = np.zeros(annots.size)
        awake  = sleep.copy()
        for iannot in uannot:
            if iannot != None:
                ann = iannot.lower()
                if 'wake' in ann or 'awake' in ann or 'pdr' in ann:
                    inds = (annots==iannot)
                    awake[inds]=1
                if 'sleep' in ann or 'spindle' in ann or 'k complex' in ann or 'sws' in ann:
                    inds = (annots==iannot)
                    sleep[inds]=1
        self.data['sleep'] = sleep
        self.data['awake'] = awake

    def state_split(self):

        self.sleep_data = self.data.loc[(self.data.sleep==1)]
        self.awake_data = self.data.loc[(self.data.awake==1)]
        return self.sleep_data,self.awake_data

    def recast(self):
        
        for icol in self.data.columns:
            try:
                self.data[icol]=self.data[icol].astype('float')
            except:
                pass

    def load_data(self,sleep_path,awake_path):

        self.sleep_data = PD.read_pickle(sleep_path)
        self.awake_data = PD.read_pickle(awake_path)
        return self.sleep_data,self.awake_data

    def call_stats(self):

        data_analysis.plot_raw_distributions(self)

def merge_pickles(path,outpath):
    files   = glob.glob(path)
    DF_list = []
    for ifile in files:
        print('Reading in %s' %(ifile))
        iDF = PD.read_pickle(ifile)
        DF_list.append(iDF)
    DF = PD.concat(DF_list)
    DF.to_pickle(outpath)

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.")
    parser.add_argument("--outdir", default="./", help="Path to output directory.")
 
    inputtype_group = parser.add_mutually_exclusive_group()
    inputtype_group.add_argument("--infile", help="Path to input file.")
    inputtype_group.add_argument("--wildcard", help="Wildcard path to merge files together.")
    args = parser.parse_args()

    # Merge files as needed
    if args.wildcard != None:
        args.infile = args.outdir+"merged_data.pickle"
        merge_pickles(args.wildcard, args.infile)

    # Save or load the sleep and awake dataframes as needed
    sleep_path = args.outdir+"sleep.pickle"
    awake_path = args.outdir+"awake.pickle"
    if not os.path.exists(sleep_path) or not os.path.exists(awake_path):
        # Initial class load and some meta variables
        DL       = data_loader(args.infile,args.outdir)
        channels = DL.get_channels()
        DL.get_state()
        DL.recast()
        DF_sleep,DF_awake = DL.state_split()
        DF_sleep.to_pickle(sleep_path)
        DF_awake.to_pickle(awake_path)
    else:
        # Initial class load and some meta variables
        DL       = data_loader(sleep_path,args.outdir)
        channels = DL.get_channels()
        DF_sleep,DF_awake = DL.load_data(sleep_path,awake_path)

    # Make statistic plots
    DL.call_stats()
    DL.plot_raw_distributions()