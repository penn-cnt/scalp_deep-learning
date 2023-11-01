# Basic Python Imports
import sys
import time
import seaborn
import argparse
import tkinter as tk
from glob import glob
import matplotlib.pyplot as PLT

# Local imports
from modules.data_loader import *
from modules.channel_clean import *
from modules.channel_mapping import *
from modules.channel_montage import *

#################
#### Classes ####
#################

class data_viewer:

    def __init__(self,infile):
        
        # Save the input info
        self.infile = infile
        self.fname  = infile.split('/')[-1]

        # Get the approx screen dimensions
        root        = tk.Tk()
        self.height = 0.9*root.winfo_screenheight()/100
        self.width  = 0.9*root.winfo_screenwidth()/100
        root.destroy()

        # Save event driven variables
        self.xlim    = []
        self.drawn_y = []

    def data_prep(self):

        # Create pointers to the relevant classes
        DL    = data_loader()
        CHCLN = channel_clean()
        CHMAP = channel_mapping()
        CHMON = channel_montage()

        # Get the raw data and pointers
        DF,self.fs = DL.direct_inputs(self.infile,'edf')

        # Get the cleaned channel names
        clean_channels = CHCLN.direct_inputs(DF.columns)
        channel_dict   = dict(zip(DF.columns,clean_channels))
        DF.rename(columns=channel_dict,inplace=True)

        # Get the channel mapping
        channel_map = CHMAP.direct_inputs(DF.columns,"HUP1020")
        DF          = DF[channel_map]

        # Get the montage
        self.DF = CHMON.direct_inputs(DF,"HUP1020")

    def montage_plot(self):
        
        # Get the number of channels to plot
        nchan = self.DF.columns.size

        # Set the label shift. 72 points equals ~1 inch in pyplot
        width_frac = (0.025*self.width)
        npnt       = int(72*width_frac)
        
        # Create the plotting environment
        fig           = PLT.figure(dpi=100,figsize=(self.width,self.height))
        gs            = fig.add_gridspec(nchan, 1, hspace=0)
        self.ax_dict  = {}
        self.lim_dict = {}
        for idx,ichan in enumerate(self.DF.columns):
            # Define the axes
            if idx == 0:
                self.ax_dict[ichan] = fig.add_subplot(gs[idx, 0])
                self.refkey         = ichan
            else:
                self.ax_dict[ichan] = fig.add_subplot(gs[idx, 0],sharex=self.ax_dict[self.refkey])

            # Get the data stats 
            idata,ymin,ymax = self.get_stats(ichan)
            xvals           = np.arange(idata.size)/self.fs
            self.xlim_orig  = [xvals[0],xvals[-1]]

            # Plot the data
            nstride = 2
            self.ax_dict[ichan].plot(xvals[::nstride],idata[::nstride],color='k')
            self.ax_dict[ichan].set_ylim([ymin,ymax])
            self.lim_dict[ichan] = [ymin,ymax]
            
            # Clean up the plot
            xticklabels = self.ax_dict[ichan].get_xticklabels().copy()
            self.ax_dict[ichan].set_yticklabels([])
            self.ax_dict[ichan].set_ylabel(ichan,fontsize=12,rotation=0,labelpad=npnt)
        
        # Add an xlabel to the final object
        self.ax_dict[ichan].set_xticklabels(xticklabels)
        self.ax_dict[ichan].set_xlabel("Time (s)",fontsize=14)

        self.ax_dict[self.refkey].set_title(self.fname,fontsize=14)
        fig.tight_layout()
        fig.canvas.mpl_connect('button_press_event', self.on_click)
        fig.canvas.mpl_connect('key_press_event', self.update_plot)
        PLT.show()

    def enlarged_plot(self,channel):
        
        # Get the data view
        idata,ymin,ymax = self.get_stats(channel)
        xvals           = np.arange(idata.size)/self.fs

        # Plot the enlarged view
        fig     = PLT.figure(dpi=100,figsize=(self.width,self.height))
        self.ax_enl = fig.add_subplot(111)
        self.ax_enl.plot(xvals,idata,color='k')
        self.ax_enl.set_xlabel("Time (s)",fontsize=14)
        self.ax_enl.set_ylabel(channel,fontsize=14)
        PLT.title(self.fname,fontsize=14)
        fig.tight_layout()
        PLT.show()

    ##########################
    #### Helper functions ####
    ##########################

    def get_stats(self,ichan):

        idata  = self.DF[ichan].values
        median = np.median(idata)
        stdev  = np.std(idata)
        idata -= median
        ymin   = -5*stdev
        ymax   = 5*stdev
        return idata,ymin,ymax

    def yscaling(self,ikey,dy):

        # Get the limits of the current plot for rescaling and recreating
        xlim      = self.ax_dict[ikey].get_xlim()
        ylim      = self.ax_dict[ikey].get_ylim()

        # Get the approximate new scale
        scale     = ylim[1]-ylim[0]
        ymin      = ylim[0]+dy*scale
        ymax      = ylim[1]-dy*scale

        # Get the data and limits with a good vertical offset
        vals      = self.DF[ikey].values
        inds      = (vals>=ymin)&(vals<=ymax)
        vals      = vals[inds]
        offset    = np.median(vals)
        vals     -= offset
        ymin     -= offset
        ymax     -= offset

        # Generate new limits
        self.ax_dict[ikey].set_ylim([ymin,ymax])

    ################################
    #### Event driven functions ####
    ################################

    def on_click(self,event):
        if event.button == 1:  # Check if left mouse button is clicked
            for ikey in self.ax_dict.keys():
                self.drawn_y.append(self.ax_dict[ikey].axvline(event.xdata, color='red', linestyle='--'))
            PLT.draw()  # Redraw the plot to update the display

            # Update the event driven zoom object
            self.xlim.append(event.xdata)

    def update_plot(self,event):
        if event.key == 'z' and len(self.xlim) == 2:
            
            # Set the xlimits
            self.ax_dict[self.refkey].set_xlim(self.xlim)
            self.xlim = []

            # Remove the vertical lines on the plot
            for iobj in self.drawn_y:
                iobj.remove()
            self.draw_y = []
        elif event.key == 'r':
            self.ax_dict[self.refkey].set_xlim(self.xlim_orig)
            self.xlim = []
        elif event.key == '+':
            for ikey in self.ax_dict.keys():
                self.yscaling(ikey,0.1)
        elif event.key == '-':
            for ikey in self.ax_dict.keys():
                self.yscaling(ikey,-0.1)
        elif event.key == '0':
            for ikey in self.ax_dict.keys():
                self.ax_dict[ikey].set_ylim(self.lim_dict[ikey])
        elif event.key == 'e':
            for ikey in self.ax_dict.keys():
                if event.inaxes == self.ax_dict[ikey]:
                    self.enlarged_plot(ikey)
        PLT.draw()


class CustomFormatter(argparse.HelpFormatter):
    """
    Custom formatting class to get a better argument parser help output.
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)

#####################
#### Helper Fncs ####
#####################

def make_help_str(idict):
    """
    Make a well-formated help string for the possible keyword mappings

    Args:
        idict (dict): Dictionary containing the allowed keywords values and their explanation.

    Returns:
        str: Formatted help string
    """

    return "\n".join([f"{key:15}: {value}" for key, value in idict.items()])

if __name__ == '__main__':

    # Argument option creation
    allowed_channel_args    = {'HUP1020': "Channels associated with a 10-20 montage performed at HUP.",
                               'RAW': "Use all possible channels. Warning, channels may not match across different datasets."}
    allowed_montage_args    = {'HUP1020': "Use a 10-20 montage.",
                               'COMMON_AVERAGE': "Use a common average montage."}
    
    # Pretty argument string creation
    allowed_channel_help   = make_help_str(allowed_channel_args)
    allowed_montage_help   = make_help_str(allowed_montage_args)

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.", formatter_class=CustomFormatter)

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--file", type=str, help="Input file to plot.")
    input_group.add_argument("--wildcard", type=str, help="Wildcard enabled path to plot multiple datasets.")

    prep_group = parser.add_argument_group('Data preparation options')
    prep_group.add_argument("--channel_list", choices=list(allowed_channel_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_channel_help}")

    args = parser.parse_args()

    # Create the file list to read in
    if args.file != None:
        files = [args.file]
    else:
        files = glob(args.wildcard)

    # Get the data prepared for viewing
    for ifile in files:
        DV = data_viewer(ifile)
        DV.data_prep()
        DV.montage_plot()
