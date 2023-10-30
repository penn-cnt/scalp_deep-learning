import os
import glob
import argparse
import numpy as np
import pandas as PD
from sys import exit,path
from pyedflib import highlevel
import matplotlib.pyplot as PLT
from scipy.signal import find_peaks
import matplotlib.colors as mcolors
from prompt_toolkit.completion import PathCompleter

# Import channel cleaning libraries
path.append("../SCALP_CONCAT-EEG/")
from modules.channel_clean import channel_clean as CC
from modules.channel_montage import channel_montage as CM

class data_analysis:

    def __init__(self):

        pass

    def peaks(self, vals, prominence=1, width=3, height=None):

        if height == None:
            height = 0.1*max(vals)

        return find_peaks(vals, prominence=prominence, width=width, height=height)
    
    def histogram_data(self,awake_alpha_channel,awake_delta_channel,sleep_alpha_channel,sleep_delta_channel):

        # Make a better plotting baseline
        self.logbins = np.logspace(7,12,50)
        self.log_x   = (0.5*(self.logbins[1:]+self.logbins[:-1]))

        # Get the histogram counts
        self.awake_alpha_cnt = np.histogram(awake_alpha_channel,bins=self.logbins)[0]
        self.awake_delta_cnt = np.histogram(awake_delta_channel,bins=self.logbins)[0]
        self.sleep_alpha_cnt = np.histogram(sleep_alpha_channel,bins=self.logbins)[0]
        self.sleep_delta_cnt = np.histogram(sleep_delta_channel,bins=self.logbins)[0]

        # Get the peak information
        self.awake_alpha_peaks, self.awake_alpha_properties = self.peaks(self.awake_alpha_cnt)
        self.awake_delta_peaks, self.awake_delta_properties = self.peaks(self.awake_delta_cnt)
        self.sleep_alpha_peaks, self.sleep_alpha_properties = self.peaks(self.sleep_alpha_cnt)
        self.sleep_delta_peaks, self.sleep_delta_properties = self.peaks(self.sleep_delta_cnt)

        # Convert peaks into the right units for plotting
        self.awake_alpha_peaks_x = self.log_x[self.awake_alpha_peaks]
        self.awake_delta_peaks_x = self.log_x[self.awake_delta_peaks]
        self.sleep_alpha_peaks_x = self.log_x[self.sleep_alpha_peaks]
        self.sleep_delta_peaks_x = self.log_x[self.sleep_delta_peaks]
        self.awake_alpha_peaks_y = self.awake_alpha_cnt[self.awake_alpha_peaks]
        self.awake_delta_peaks_y = self.awake_delta_cnt[self.awake_delta_peaks]
        self.sleep_alpha_peaks_y = self.sleep_alpha_cnt[self.sleep_alpha_peaks]
        self.sleep_delta_peaks_y = self.sleep_delta_cnt[self.sleep_delta_peaks]

        self.awake_alpha_properties["left_ips"]  = [self.log_x[int(np.floor(ival))] for ival in self.awake_alpha_properties["left_ips"]]
        self.awake_delta_properties["left_ips"]  = [self.log_x[int(np.floor(ival))] for ival in self.awake_delta_properties["left_ips"]]
        self.sleep_alpha_properties["left_ips"]  = [self.log_x[int(np.floor(ival))] for ival in self.sleep_alpha_properties["left_ips"]]
        self.sleep_delta_properties["left_ips"]  = [self.log_x[int(np.floor(ival))] for ival in self.sleep_delta_properties["left_ips"]]
        self.awake_alpha_properties["right_ips"] = [self.log_x[int(np.ceil(ival))] for ival in self.awake_alpha_properties["right_ips"]]
        self.awake_delta_properties["right_ips"] = [self.log_x[int(np.ceil(ival))] for ival in self.awake_delta_properties["right_ips"]]
        self.sleep_alpha_properties["right_ips"] = [self.log_x[int(np.ceil(ival))] for ival in self.sleep_alpha_properties["right_ips"]]
        self.sleep_delta_properties["right_ips"] = [self.log_x[int(np.ceil(ival))] for ival in self.sleep_delta_properties["right_ips"]]

        # Assign colors for easy pairing visually later
        self.awake_alpha_properties['color'] = self.color_names[:len(self.awake_alpha_peaks)]
        self.awake_delta_properties['color'] = self.color_names[:len(self.awake_delta_peaks)]
        self.sleep_alpha_properties['color'] = self.color_names[:len(self.sleep_alpha_peaks)]
        self.sleep_delta_properties['color'] = self.color_names[:len(self.sleep_delta_peaks)]

    def read_clean_data(self,file):

        # Read in data
        self.rawdata, raw_channel_metadata,_ = highlevel.read_edf(file)

        # Get the sampling frequency
        self.fs[file] = np.array([ichannel['sample_frequency'] for ichannel in raw_channel_metadata])[0]

        # Get the raw channels
        channels   = [ival['label'] for ival in raw_channel_metadata]
        CC_handler = CC()
        channels   = CC_handler.direct_inputs(channels)

        # Make the data into dataframe
        self.rawdata = PD.DataFrame(self.rawdata.T,columns=channels)

        # Get the montages version of the data
        CM_handler   = CM()
        self.rawdata = CM_handler.direct_inputs(self.rawdata,"HUP1020")

    def get_timeseries(self,newfile):

        # Determine if we need to read in the data again
        try:
            if newfile == self.oldfile:
                pass
            else:
                self.oldfile = newfile
                self.read_clean_data(newfile)

        except AttributeError:
            self.oldfile = newfile
            self.read_clean_data(newfile)

    def update_data_dict(self,current_meta,meta_str):

        for imeta in current_meta.values:
            file   = imeta[0]
            time_0 = imeta[1]
            time_1 = imeta[2]
            self.get_timeseries(file)
            if file not in list(self.data_dict.keys()):
                self.data_dict[file] = {}
                self.data_dict[file]['raw'] = self.rawdata
            self.data_dict[file][meta_str]    = np.chararray(self.data_dict[file]['raw'].shape,unicode=True)
            self.data_dict[file][meta_str][:] = 'k'

    def update_data_colors(self,idf,ichannel,channel_data,props,meta_str):
        
        for ii in range(len(props['left_ips'])):
            inds  = (channel_data>=props['left_ips'][ii])&(channel_data<props['right_ips'][ii])
            slice = idf.iloc[inds] 
            color = props["color"][ii]

            for irow in range(slice.shape[0]):
                datarow    = slice.iloc[irow]
                samp_start = int(self.fs[datarow.file]*datarow.t_start)
                samp_end   = int(self.fs[datarow.file]*datarow.t_end)
                icol       = np.argwhere(self.data_dict[datarow.file]['raw'].columns==ichannel)[0][0]
                self.data_dict[datarow.file][meta_str][samp_start:samp_end,icol] = color


    def plot_handler(self):

        # First split according to the unique id
        uids = np.intersect1d(self.sleep_data['uid'].unique(),self.awake_data['uid'].unique())
        for i_uid in uids:

            # Get the awake and sleep splits
            awake_df = self.awake_data.loc[(self.awake_data.uid==i_uid)]
            sleep_df = self.sleep_data.loc[(self.sleep_data.uid==i_uid)]

            # Get the alpha and delta dataslices
            awake_alpha = awake_df.loc[(awake_df.tag==self.alpha_str)]
            awake_delta = awake_df.loc[(awake_df.tag==self.delta_str)]
            sleep_alpha = sleep_df.loc[(sleep_df.tag==self.alpha_str)]
            sleep_delta = sleep_df.loc[(sleep_df.tag==self.delta_str)]

            # Get associated files and times for each group
            self.awake_alpha_meta = awake_alpha[['file','t_start','t_end']]
            self.awake_delta_meta = awake_delta[['file','t_start','t_end']]
            self.sleep_alpha_meta = sleep_alpha[['file','t_start','t_end']]
            self.sleep_delta_meta = sleep_delta[['file','t_start','t_end']]

            # Get the timeseries loaded into memory
            self.data_dict = {}
            self.fs        = {}
            self.update_data_dict(self.awake_alpha_meta,'awake_alpha')
            self.update_data_dict(self.awake_delta_meta,'awake_delta')
            self.update_data_dict(self.sleep_alpha_meta,'sleep_alpha')
            self.update_data_dict(self.sleep_delta_meta,'sleep_delta')

            # Loop over the channels and make plots
            for ichannel in self.channels:

                # Get channel values of the spectral energy
                awake_alpha_channel = awake_alpha[ichannel].values
                awake_delta_channel = awake_delta[ichannel].values
                sleep_alpha_channel = sleep_alpha[ichannel].values
                sleep_delta_channel = sleep_delta[ichannel].values

                # Get the histogram stats and find peaks
                self.histogram_data(awake_alpha_channel,awake_delta_channel,sleep_alpha_channel,sleep_delta_channel)

                # Loop over the different peaks to populate the masks
                prop_1 = self.awake_alpha_properties
                prop_2 = self.awake_delta_properties
                prop_3 = self.sleep_alpha_properties
                prop_4 = self.sleep_delta_properties
                self.update_data_colors(awake_alpha,ichannel,awake_alpha_channel,prop_1,'awake_alpha')
                self.update_data_colors(awake_delta,ichannel,awake_delta_channel,prop_2,'awake_delta')
                self.update_data_colors(sleep_alpha,ichannel,sleep_alpha_channel,prop_3,'sleep_alpha')
                self.update_data_colors(sleep_delta,ichannel,sleep_delta_channel,prop_4,'sleep_delta')

                # Make plots
                self.plot_raw_distributions(ichannel,i_uid,awake_alpha_channel,awake_delta_channel,sleep_alpha_channel,sleep_delta_channel)
            self.plot_timeseries(ichannel,i_uid)

    def plot_timeseries(self,ichannel,i_uid):
        
        # Get the number of timeseries files we need to plot
        filenames = list(self.data_dict.keys())
        for ii,ifile in enumerate(filenames):
            filebase = ifile.split('/')[-1]
            for idx,ichannel in enumerate(self.channels):
                y             = self.data_dict[ifile]['raw'][ichannel].values
                x             = np.arange(y.size)
                try:
                    c_awake_alpha = list(self.data_dict[ifile]['awake_alpha'][:,idx])
                except KeyError:
                    c_awake_alpha = 'k'
                
                try:
                    c_awake_delta = list(self.data_dict[ifile]['awake_delta'][:,idx])
                except KeyError:
                    c_awake_delta = 'k'
                
                try:
                    c_sleep_alpha = list(self.data_dict[ifile]['sleep_alpha'][:,idx])
                except KeyError:
                    c_sleep_alpha = 'k'
                
                try:
                    c_sleep_delta = list(self.data_dict[ifile]['sleep_delta'][:,idx])
                except KeyError:
                    c_sleep_delta = 'k'

                # Make the plot
                fig = PLT.figure(dpi=100,figsize=(12.,8.))
                ax1 = fig.add_subplot(411)
                ax2 = fig.add_subplot(412,sharex=ax1)
                ax3 = fig.add_subplot(413,sharex=ax1)
                ax4 = fig.add_subplot(414,sharex=ax1)
                ax1.scatter(x,y,color=c_awake_alpha,s=1)
                ax2.scatter(x,y,color=c_awake_delta,s=1)
                ax3.scatter(x,y,color=c_sleep_alpha,s=1)
                ax4.scatter(x,y,color=c_sleep_delta,s=1)
                ax1.set_ylabel(['Alpha Awake'])
                ax2.set_ylabel(['Alpha Delta'])
                ax3.set_ylabel(['Sleep Alpha'])
                ax4.set_ylabel(['Sleep Delta'])
                ax1.set_xticklabels([])
                ax2.set_xticklabels([])
                ax3.set_xticklabels([])
                ax1.set_title("Patient id: %03d, Channel: %s, File:%s" %(i_uid,ichannel,filebase), fontsize=13)
                fig.tight_layout()
                PLT.savefig(self.plotdir+"%03d_%s_timeseries_%02d.png" %(i_uid,ichannel,ii))
                PLT.close("all")

    def plot_raw_distributions(self,ichannel,i_uid,awake_alpha_channel,awake_delta_channel,sleep_alpha_channel,sleep_delta_channel):
                
        # Make the plot canvas
        fig = PLT.figure(dpi=100,figsize=(10.,8.))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222,sharex=ax1)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224,sharex=ax3)

        # Histograms
        ax1.hist(awake_alpha_channel,bins=self.logbins,histtype='step',color='k',label="Awake Alpha")
        ax2.hist(awake_delta_channel,bins=self.logbins,histtype='step',color='k',label="Awake Delta")
        ax3.hist(sleep_alpha_channel,bins=self.logbins,histtype='step',color='k',label="Sleep Alpha")
        ax4.hist(sleep_delta_channel,bins=self.logbins,histtype='step',color='k',label="Sleep Delta")
        
        # Peak finder objects vertical
        ax1.vlines(x=self.awake_alpha_peaks_x, ymin=0, ymax = self.awake_alpha_peaks_y, color = self.awake_alpha_properties['color'])
        ax2.vlines(x=self.awake_delta_peaks_x, ymin=0, ymax = self.awake_delta_peaks_y, color = self.awake_delta_properties['color'])
        ax3.vlines(x=self.sleep_alpha_peaks_x, ymin=0, ymax = self.sleep_alpha_peaks_y, color = self.sleep_alpha_properties['color'])
        ax4.vlines(x=self.sleep_delta_peaks_x, ymin=0, ymax = self.sleep_delta_peaks_y, color = self.sleep_delta_properties['color'])

        # Peak finder objects horizontal
        y1 = self.awake_alpha_properties["width_heights"]
        y2 = self.awake_delta_properties["width_heights"]
        y3 = self.sleep_alpha_properties["width_heights"]
        y4 = self.sleep_delta_properties["width_heights"]
        xmin1 = self.awake_alpha_properties["left_ips"]
        xmin2 = self.awake_delta_properties["left_ips"]
        xmin3 = self.sleep_alpha_properties["left_ips"]
        xmin4 = self.sleep_delta_properties["left_ips"]
        xmax1 = self.awake_alpha_properties["right_ips"]
        xmax2 = self.awake_delta_properties["right_ips"]
        xmax3 = self.sleep_alpha_properties["right_ips"]
        xmax4 = self.sleep_delta_properties["right_ips"]
        ax1.hlines(y=y1, xmin=xmin1, xmax=xmax1, color = self.awake_alpha_properties['color'])
        ax2.hlines(y=y2, xmin=xmin2, xmax=xmax2, color = self.awake_delta_properties['color'])
        ax3.hlines(y=y3, xmin=xmin3, xmax=xmax3, color = self.sleep_alpha_properties['color'])
        ax4.hlines(y=y4, xmin=xmin4, xmax=xmax4, color = self.sleep_delta_properties['color'])

        # Final plot prep
        ax1.legend(loc=1)
        ax2.legend(loc=1)
        ax3.legend(loc=1)
        ax4.legend(loc=1)
        ax1.set_xscale('log')
        ax2.set_xscale('log')
        ax3.set_xscale('log')
        ax4.set_xscale('log')
        PLT.title("Patient id: %03d, Channel: %s" %(i_uid,ichannel), fontsize=13)
        fig.tight_layout()
        PLT.savefig(self.plotdir+"%03d_%s.png" %(i_uid,ichannel))
        PLT.close("all")

class data_loader(data_analysis):

    def __init__(self,infile, outdir):

        # Define any hard-coded variables here, or save passed arguments to class
        self.outdir    = outdir
        self.alpha_str = '[8.0,12.0]'
        self.delta_str = '[1.0,4.0]'

        # Read in the input dataset and get channel names
        self.data = PD.read_pickle(infile)
        self.get_channels()

        # Create output directory structure as needed
        self.plotdir   = self.outdir+"PLOTS/"
        if not os.path.exists(self.plotdir):
            os.system("mkdir -p %s" %(self.plotdir))

        # Define the colorset
        color_names      = np.array(list(mcolors.BASE_COLORS.keys()))
        self.color_names = []
        for icolor in color_names:
            if icolor not in ['k','w','y']:
                self.color_names.append(icolor)
        self.color_names = np.array(self.color_names)
    
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

        data_analysis.plot_handler(self)

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
    parser = argparse.ArgumentParser(description="Analysis tool for sleep/awake data.")
    parser.add_argument("--outdir", default="./", help="Path to output directory.")
    parser.add_argument("--awakefile", help="Filepath to a csv with outputs from SCALP_CONCAT_EEG that have already been sliced against awake annotations.")
    parser.add_argument("--sleepfile", help="Filepath to a csv with outputs from SCALP_CONCAT_EEG that have already been sliced against sleep annotations.")
 
    inputtype_group = parser.add_mutually_exclusive_group()
    inputtype_group.add_argument("--infile", help="Path to input pickle file. Either individual outputs from SCALP_CONCAT_EEG or a merged pickle.")
    inputtype_group.add_argument("--wildcard", help="Wildcard path to merge SCALP_CONCAT_EEG pickle files together.")
    args = parser.parse_args()

    # Merge files as needed
    if args.wildcard != None:

        # Tab completion enabled input
        completer = PathCompleter()
        print("Please enter an output path for the merged dataset.")
        print("If left blank, defaults to %smerged_data.pickle")
        file_path = prompt("Please enter path: ", completer=completer)

        if file_path == '':
            args.infile = args.outdir+"merged_data.pickle"
        else:
            args.infile = file_path    
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