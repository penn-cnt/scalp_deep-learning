# User interface imports
import pyautogui
import numpy as np

# Matplotlib import and settings
import matplotlib.pyplot as PLT
from matplotlib.ticker import MultipleLocator

# Local Imports
from components.internal.data_handler import *
from components.internal.event_handler import *
from components.internal.observer_handler import *

class data_viewer(Subject,event_handler):

    def __init__(self, infile, args, tight_layout_dict):
        
        # Save the input info
        self.infile            = infile
        self.fname             = infile.split('/')[-1]
        self.args              = args
        self.tight_layout_dict = tight_layout_dict

        # Get the approx screen dimensions and set some plot variables
        self.height  = 0.9*pyautogui.size().height/100
        self.width   = 0.9*pyautogui.size().width/100
        self.supsize = self.fontsize_scaler(16,14,self.width)
        self.supsize = np.min([self.supsize,16])

        # Prepare the data
        DH            = data_handler(args,infile)
        rawobj        = DH.workflow()
        self.DF       = rawobj[0]
        self.fs       = rawobj[1]
        self.t_max    = rawobj[2]
        self.duration = rawobj[3]
        self.t0       = rawobj[4]

    def workflow(self):

        # Attach the observers
        self.attach_objects()

        # Make the initial plot info
        self.create_plot_info()

        # Draw the plot for the first time
<<<<<<< Updated upstream
        self.draw_plots()
=======
        self.draw_base_plots()
>>>>>>> Stashed changes

    def attach_objects(self):
        """
        Attach observers here so we can have each multiprocessor see the pointers correctly.
        """

        # Create the observer objects
        self._event_observers = []

        # Attach observers
        self.add_event_observer(event_observer)

    def create_plot_info(self):

        # Store some valuable information about the plot to reference for events and modifications
<<<<<<< Updated upstream
        self.plot_info              = {}
        self.plot_info['axes']      = {}
        self.plot_info['ylim']      = {}
        self.plot_info['shade']     = {}
        self.plot_info['xlim_orig'] = [self.t0,self.t0+self.duration]
        self.plot_info['xvals']     = np.arange(self.DF.shape[0])/self.fs

    def draw_plots(self):
=======
        self.plot_info                = {}
        self.plot_info['axes']        = {}
        self.plot_info['ylim']        = {}
        self.plot_info['shade']       = {}
        self.plot_info['xlim_orig']   = [self.t0,self.t0+self.duration]
        self.plot_info['xvals']       = np.arange(self.DF.shape[0])/self.fs
        self.plot_info['zoom_lines']  = []
        self.plot_info['annot_lines'] = []
        self.plot_info['annot_value'] = []

    ############################
    #### Plotting functions ####
    ############################    

    def draw_base_plots(self):
>>>>>>> Stashed changes

        # Set the label shift. 72 points equals ~1 inch in pyplot
        width_frac = (0.025*self.width)
        npnt       = int(72*width_frac)
        
        # Create the plotting environment
        nrows           = len(self.DF.columns)
        self.fig        = PLT.figure(dpi=100,figsize=(self.width,self.height))
        gs              = self.fig.add_gridspec(nrows, 1, hspace=0)
        for idx,ichan in enumerate(self.DF.columns):
            # Define the axes
            if idx == 0:
                self.plot_info['axes'][ichan] = self.fig.add_subplot(gs[idx, 0])
                self.first_chan               = ichan
            else:
                self.plot_info['axes'][ichan] = self.fig.add_subplot(gs[idx, 0],sharex=self.plot_info['axes'][self.first_chan])

            # Get the data stats 
            idata,ymin,ymax = self.get_stats(ichan)

            # Plot the data
            self.plot_info['axes'][ichan].plot(self.plot_info['xvals'][::self.args.nstride],idata[::self.args.nstride],color='k')
            self.plot_info['axes'][ichan].set_ylim([ymin,ymax])
            self.plot_info['ylim'][ichan] = [ymin,ymax]

            # Add in shading for the original axes limits
            self.plot_info['shade'][ichan] = self.plot_info['axes'][ichan].axvspan(self.plot_info['xlim_orig'][0], self.plot_info['xlim_orig'][1], facecolor='orange',alpha=0.2)

            # Clean up the plot
            for label in self.plot_info['axes'][ichan].get_xticklabels():
                label.set_alpha(0)
            self.plot_info['axes'][ichan].set_yticklabels([])
            self.plot_info['axes'][ichan].set_ylabel(ichan,fontsize=12,rotation=0,labelpad=npnt)
            self.plot_info['axes'][ichan].xaxis.grid(True)
        
        # X-axis cleanup
        self.last_chan = ichan
        self.plot_info['axes'][ichan].set_xlim(self.plot_info['xlim_orig'])

        # Add an xlabel to the final object
        self.plot_info['axes'][self.last_chan].xaxis.set_major_locator(MultipleLocator(1))
        self.plot_info['axes'][self.last_chan].set_xlabel("Time (s)",fontsize=14)
        for label in self.plot_info['axes'][self.last_chan].get_xticklabels():
            label.set_alpha(1)

        # Set the axes title object
        self.generate_title_str()
        self.plot_info['axes'][self.first_chan].set_title(self.title_str,fontsize=10)
        
        # Set the figure title object
        self.generate_suptitle_str()
        PLT.suptitle(self.suptitle,fontsize=self.supsize)
        
        # Layout handling using previous plot layout or find it for the first time
        if self.tight_layout_dict == None:
            self.fig.tight_layout()
        else:
            self.fig.subplots_adjust(**self.tight_layout_dict)
        
        # Event associations
        self.fig.canvas.mpl_connect('button_press_event', self.notify_event_observers)
        self.fig.canvas.mpl_connect('key_press_event', self.notify_event_observers)

        # Show the results
        PLT.show()

        # Store and return tight layout params for faster subsequent plots
        if self.tight_layout_dict == None:
            self.tight_layout_dict = {par : getattr(self.fig.subplotpars, par) for par in ["left", "right", "bottom", "top", "wspace", "hspace"]}
        return self.tight_layout_dict

<<<<<<< Updated upstream
=======
    def draw_annotations(self,xpos,annotation,ichannel):

        for ikey in self.plot_info['axes'].keys():
            self.plot_info['annot_lines'].append(self.plot_info['axes'][ikey].axvline(xpos, color='g', linestyle='--'))

        # Add the annotation
        ymin,ymax = self.plot_info['axes'][ichannel].get_ylim()
        ypos      = 0.5*(ymin+ymax)
        self.plot_info['axes'][ichannel].annotate(text=annotation, xy =(xpos,ypos),bbox=dict(boxstyle="round", facecolor="gray", alpha=0.7))

>>>>>>> Stashed changes
    def enlarged_plot(self,channel):
        
        # Get the data view
        idata,ymin,ymax = self.get_stats(channel)
        xvals           = np.arange(idata.size)/self.fs

        # Get the current limits of the main viewer
        xlims = self.ax_dict[self.first_chan].get_xlim()
        ylims = [ymin,ymax]

        # Plot the enlarged view
        fig     = PLT.figure(dpi=100,figsize=(self.width,self.height))
        self.ax_enl = fig.add_subplot(111)
        self.ax_enl.plot(xvals,idata,color='k')
        self.ax_enl.set_xlabel("Time (s)",fontsize=14)
        self.ax_enl.set_ylabel(channel,fontsize=14)
        self.ax_enl.set_xlim(xlims)
        self.ax_enl.set_ylim(ylims)
        PLT.title(self.fname,fontsize=14)
        fig.tight_layout()
        PLT.show()

    ##########################
    #### Helper functions ####
    ##########################

    def fontsize_scaler(self,font_ref,width_ref,width_val):
        return font_ref+2*np.floor((width_val-width_ref))

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

    def generate_title_str(self):
        upa        = u'\u2191'  # Up arrow
        downa      = u'\u2193'  # Down arrow
        lefta      = u'\u2190'  # Left arrow
        righta     = u'\u2192'  # Right arrow
        self.title_str  = r"z=Zoom between mouse clicks; 'r'=reset x-scale; 'x'=Show entire x-axis; '0'=reset y-scale; 't'=Toggle targets; 'q'=quit current plot; 'Q'=quit the program entirely"
        self.title_str += '\n'
        self.title_str += r"'%s'=Increase Gain; '%s'=Decrease Gain; '%s'=Shift Left; '%s'=Shift Right; '<'=Minor Shift Left; '>'=Minor Shift Right; 'e'=Zoom-in plot of axis the mouse is on;" %(upa, downa, lefta, righta)
        if self.args.annotations:
            self.title_str += '\n'
            self.title_str += r"1=Sleep State; 2=Spike Presence; 3=Seizure; 4=Focal Slowing; 5=Generalized Slowing; 6=Artifact Heavy"

    def generate_suptitle_str(self):

        # Base string
        self.suptitle = self.fname
        
        # If using flagging, create new string
        if self.args.annotations:
            self.suptitle += '\n'
            for ival in self.flagged_out:
                self.suptitle += f" {ival} |"
            self.suptitle = self.suptitle[:-1]

<<<<<<< Updated upstream
    def flag_toggle(self,label_name,counter_name,str_pos):
        
        # Get the labels
        labels  = getattr(self,label_name)
        counter = getattr(self,counter_name)

        # Handle the counter logic
        counter+=1
        if counter == 4 and counter_name != 'artifact_counter':
            counter = 0
        elif counter == 2 and counter_name == 'artifact_counter':
            counter = 0

        # Update the substring
        newval  = labels[counter]
        self.flagged_out[str_pos] = newval

        # Generate the new suptilte
        self.generate_suptitle_str()

        # Set the new title
        PLT.suptitle(f"{self.suptitle}",fontsize=self.supsize)

        # Set the new counter values
        setattr(self,counter_name,counter)

    def save_flag_state(self):

        # Create output column list
        xlims   = self.ax_dict[self.last_chan].get_xlim()
        outcols = ['filename','username','assigned_t0','assigned_t1','evaluated_t0','evaluated_t1','sleep_state','spike_state','seizure_state','focal_slowing','general_slowing','artifacts']
        outvals = [self.infile,self.args.username,self.xlim_orig[0],self.xlim_orig[1],xlims[0],xlims[1]]
        outvals = outvals+self.flagged_out

        # Make the temporary dataframe to concat to outputs
        iDF = PD.DataFrame([outvals],columns=outcols)

        # Check for file
        if path.exists(self.args.outfile):
            out_DF = PD.read_csv(self.args.outfile)
            out_DF = PD.concat((out_DF,iDF),ignore_index=True)
        else:
            out_DF = iDF

        # Save the results
        if not self.args.debug:
            out_DF = out_DF.drop_duplicates()
            out_DF.to_csv(self.args.outfile,index=False)

=======
>>>>>>> Stashed changes
    ################################
    #### Event driven functions ####
    ################################

<<<<<<< Updated upstream
    def on_click(self,event):
        """
        Click driven events for the plot object.

        Args:
            Matplotlib event.
        """

        # Left click defines the zoom ranges
        if event.button == 1:
            # Loop over the axes and draw the zoom ranges
            for ikey in self.ax_dict.keys():
                self.drawn_y.append(self.ax_dict[ikey].axvline(event.xdata, color='red', linestyle='--'))
            
            # Redraw the plot to update the display
            PLT.draw()

            # Update the event driven zoom object
            self.xlim.append(event.xdata)

=======
>>>>>>> Stashed changes
    def update_plot(self,event):
        """
        Key driven events for the plot object.

        Args:
            Matplotlib event.
        """

        # Zoom on 'z' press and when there are two bounds
        if event.key == 'z' and len(self.xlim) == 2:
            
            # Set the xlimits
            self.ax_dict[self.refkey].set_xlim(self.xlim)
            self.xlim = []

            # Remove the vertical lines on the plot
            for iobj in self.drawn_y:
                iobj.remove()
            self.draw_y = []
        # Reset the -axes of the plot
        elif event.key == 'r':
            self.ax_dict[self.refkey].set_xlim(self.xlim_orig)
            self.xlim = []
        # Increase gain
        elif event.key == 'up':
            for ikey in self.ax_dict.keys():
                self.yscaling(ikey,0.1)
        # Decrease gain
        elif event.key == 'down':
            for ikey in self.ax_dict.keys():
                self.yscaling(ikey,-0.1)
        # Shift back in time
        elif event.key == 'left':
            current_xlim = self.ax_dict[self.refkey].get_xlim()
            current_xlim = [ival-self.duration for ival in current_xlim]
            for ikey in self.ax_dict.keys():
                self.ax_dict[ikey].set_xlim(current_xlim)
        # Shift forward in time
        elif event.key == 'right':
            current_xlim = self.ax_dict[self.refkey].get_xlim()
            current_xlim = [ival+self.duration for ival in current_xlim]
            for ikey in self.ax_dict.keys():
                self.ax_dict[ikey].set_xlim(current_xlim)
        # Shift back in time
        elif event.key == '<':
            current_xlim = self.ax_dict[self.refkey].get_xlim()
            current_xlim = [ival-self.duration/2. for ival in current_xlim]
            for ikey in self.ax_dict.keys():
                self.ax_dict[ikey].set_xlim(current_xlim)
        # Shift forward in time
        elif event.key == '>':
            current_xlim = self.ax_dict[self.refkey].get_xlim()
            current_xlim = [ival+self.duration/2. for ival in current_xlim]
            for ikey in self.ax_dict.keys():
                self.ax_dict[ikey].set_xlim(current_xlim)
        # Show the entire x-axis
        elif event.key == 'x':
            for ikey in self.ax_dict.keys():
                self.ax_dict[ikey].set_xlim([0,self.t_max])
        # Reset gain
        elif event.key == '0':
            for ikey in self.ax_dict.keys():
                self.ax_dict[ikey].set_ylim(self.lim_dict[ikey])
        # Enlarge a singular plot
        elif event.key == 'e':
            for ikey in self.ax_dict.keys():
                if event.inaxes == self.ax_dict[ikey]:
                    self.enlarged_plot(ikey)
        # Show annotations
        elif event.key == 'A':
            if 'trial_type' in self.events_df.columns:
                annot_xval = self.events_df['onset'][0]
                annot_text = self.events_df['trial_type'][0]
                
                if len(self.drawn_a) == 0:
                    for ikey in self.ax_dict.keys():
                        self.drawn_a.append(self.ax_dict[ikey].axvline(annot_xval, color='blue', linestyle='--',lw=2))
                    self.annot_obj = self.ax_dict[self.last_chan].text(annot_xval, 0, annot_text,bbox=dict(boxstyle='round', facecolor='lightgray', 
                                                                    edgecolor='none', alpha=1.0),verticalalignment='center', horizontalalignment='left', fontsize=12)

                else:
                    # Remove the vertical lines on the plot
                    for iobj in self.drawn_a:
                        iobj.remove()
                    self.annot_obj.remove()
                    self.draw_a = []
        # Iterate over target dictionary if available to show mapped colors
        elif event.key == 't' and hasattr(self,'color_dict'):
            for icnt in range(len(self.t_colors)):
                ikey    = self.color_keys[icnt]
                objects = self.t_obj[ikey]
                for iobj in objects:
                    if icnt == self.color_cnt:
                        # Change visibility
                        iobj.set_visible(True)

                        # Update title
                        PLT.suptitle(self.fname+" | "+str(ikey),fontsize=self.supsize)
                    else:
                        iobj.set_visible(False)

            if self.color_cnt < self.ncolor:
                self.color_cnt += 1
            else:
                self.color_cnt = 0
                PLT.suptitle(self.fname,fontsize=self.supsize)
        # Quit functionality
        elif event.key == 'Q':
            PLT.close("all")
            sys.exit()

        # Make sure the axes colorscheme is updated
        newlim = self.ax_dict[self.last_chan].get_xlim()
        if (newlim[0] == self.xlim_orig[0]) and (newlim[1] == self.xlim_orig[1]):
            ialpha = 0.2
        else:
            ialpha = 0
        for ichan in self.DF.columns:
            self.shade_dict[ichan].set_alpha(ialpha)

        PLT.draw()