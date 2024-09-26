import pylab as PLT
from sys import exit
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# Local Imports
from components.internal.observer_handler import *
from components.internal.annotation_handler import *

class event_observer(Observer):

    def listen_event(self,event):
        
        # Event logic
        button_flag = hasattr(event,'button')
        if button_flag:
            event_handler.button_response(self,event)
        else:
            # Quit the plot
            if event.key == 'Q':
                event_handler.quit(self)
            # Enlarge subplot options
            elif event.key == 'e':
                event_handler.enlarge(self,event)
            # Annotation options
            elif event.key == 'a':
                
                # Use terminal based annotation
                event_handler.annotate(self,event)

class event_handler:

    def __init__(self):
        pass

    def button_response(self,event):

        if event.button == 1:
            pass

    #####################
    #### Key options ####
    #####################

    def quit(self):
        PLT.close("all")
        exit()

    def enlarge(self,event):

        for ikey in self.ax_dict.keys():
            if event.inaxes == self.ax_dict[ikey]:
                self.enlarged_plot(ikey)

    def annotate(self,event):

        # Read in the selection variables
        script_dir        = '/'.join(os.path.abspath(__file__).split('/')[:-3])
        self.annot_config = yaml.safe_load(open(f"{script_dir}/configs/annotation_config.yaml","r"))
        for ikey in self.annot_config.keys():self.annot_config[ikey] = dict(self.annot_config[ikey])

        # Provide the user annotation options
        print("Entering annotation mode. Q/q to quit.")
        for ikey in self.annot_config.keys():
            print(f"Enter {self.annot_config[ikey]['key']} to toggle through {ikey} annotations.")
        print("Enter all other annotations as comma separated strings.")

        # Enter in interactive mode
        #while True:
            
