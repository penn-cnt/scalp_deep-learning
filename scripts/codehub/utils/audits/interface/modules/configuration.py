import os
import dearpygui.dearpygui as dpg

class configuration_handler:

    def __init__(self):
        pass

    def showConfiguration(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.65*main_window_width)
        help_window_width  = int(0.32*main_window_width)
        
        # Get the approximate number of characters allowed per-line. One time call to self to be visible across all widgets.
        max_pixel_width  = 8
        self.nchar_child = int(child_window_width/max_pixel_width)
        self.nchar_help  = int(help_window_width/max_pixel_width)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):

                ######################### 
                ###### Input Block ######
                #########################

                # Input pathing
                with dpg.group(horizontal=True):
                    arg_var = 'input_str'
                    dpg.add_text(f"{'Configuration File Path':40}")
                    self.input_path_widget_text = dpg.add_input_text(width=int(0.35*child_window_width), default_value=f"{self.script_dir}/config/audit.md5sum.linux")
                    self.input_path_widget      = dpg.add_button(label="Select File", width=int(0.14*child_window_width), callback=lambda sender, app_data:self.init_file_selection(self.input_path_widget_text, sender, app_data))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_help(self.configuration_help, sender, app_data), tag=arg_var)

            # Text widget
            with dpg.child_window(width=help_window_width):
                with dpg.group():
                    dpg.add_text("Help:")
                    self.configuration_help = dpg.add_text("", wrap=0.95*help_window_width)