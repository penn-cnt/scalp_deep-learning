import pylab as PLT
from sys import exit
<<<<<<< Updated upstream
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
=======
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
    def button_response(self,event):
=======
    def button_response(self,event,line_container=None,line_color='r'):

>>>>>>> Stashed changes

        if event.button == 1:
            pass

    #####################
    #### Key options ####
    #####################

    def quit(self):
        PLT.close("all")
        exit()

    def enlarge(self,event):

<<<<<<< Updated upstream
        for ikey in self.ax_dict.keys():
            if event.inaxes == self.ax_dict[ikey]:
=======
        for ikey in self.plot_info['axes'].keys():
            if event.inaxes == self.plot_info['axes'][ikey]:
>>>>>>> Stashed changes
                self.enlarged_plot(ikey)

    def annotate(self,event):

        # Read in the selection variables
        script_dir        = '/'.join(os.path.abspath(__file__).split('/')[:-3])
        self.annot_config = yaml.safe_load(open(f"{script_dir}/configs/annotation_config.yaml","r"))
<<<<<<< Updated upstream
        for ikey in self.annot_config.keys():self.annot_config[ikey] = dict(self.annot_config[ikey])

        # Provide the user annotation options
        print("Entering annotation mode. Q/q to quit.")
        for ikey in self.annot_config.keys():
            print(f"Enter {self.annot_config[ikey]['key']} to toggle through {ikey} annotations.")
        print("Enter all other annotations as comma separated strings.")

        # Enter in interactive mode
        #while True:
=======

        # Do some cleanup on the input options
        for ikey in self.annot_config.keys():self.annot_config[ikey] = dict(self.annot_config[ikey])
        for ikey in list(self.annot_config.keys()):
            self.annot_config[ikey.lower()] = self.annot_config.pop(ikey)

        # Print the options for the user.
        if len(self.plot_info['annot_lines']) == 0:
            print("Available options:")
            for ikey in self.annot_config.keys():print(f"    - {ikey}")
            print("Inputs not in this list will be taken as your annotation.")

        # Query the user for information
        initial_options = list(self.annot_config.keys())
    
        # Setup auto-completion for the prompt
        options_completer = WordCompleter(initial_options)
        
        # Ask user to choose one of the options
        selected_option = prompt('Please choose an option (Q/q to quit): ', completer=options_completer)

        # Check for annotations in the config to give specific answers
        if selected_option.lower() in self.annot_config.keys():
            
            # get the current option list and display for user
            new_options       = list(self.annot_config[selected_option.lower()]['options'])
            print("Available options:")
            for ival in new_options:
                print(f"    - {ival}")

            options_completer = WordCompleter(new_options)
            selected_value    = prompt('Please choose an option: ', completer=options_completer)
            annotation        = f"{selected_option}_{selected_value}"
        else:
            annotation = selected_option

        # Get the channel the user clicked on
        for ikey in self.plot_info['axes'].keys():
            if event.inaxes == self.plot_info['axes'][ikey]:
                ichannel = ikey

        # Draw the annotation line at the event location
        self.draw_annotations(event.xdata,annotation,ichannel)
            
        # Redraw the plot to update the display
        PLT.draw()

>>>>>>> Stashed changes
            
