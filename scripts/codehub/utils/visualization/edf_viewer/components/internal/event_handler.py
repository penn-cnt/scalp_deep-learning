import pylab as PLT
from sys import exit
from multiprocessing import Process,Queue
import threading

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

        # Spawn a new process to avoid conflicts with two tkinter windows
        annot_main()
