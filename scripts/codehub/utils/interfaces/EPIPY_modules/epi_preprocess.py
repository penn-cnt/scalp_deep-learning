import dearpygui.dearpygui as dpg
from modules import preprocessing
from configs.makeconfigs import *

import modules.addons.preprocessing as PP

class epi_preprocess_handler:

    def __init__(self):
        pass

    def showPreprocess(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.66*main_window_width)
        help_window_width  = int(0.33*main_window_width)

        # Get the module info
        MC         = make_config(preprocessing,None)
        method_str = MC.print_methods(silent=True)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):
                pass

            # Text widget
            with dpg.group():
                dpg.add_text("Preprocessing Options:")
                self.preprocess_help = dpg.add_text(method_str)