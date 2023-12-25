import dearpygui.dearpygui as dpg

class dataimport_handler:

    def __init__(self):
        pass

    def showDataImport(self, main_window_width = 1280):

        # Child Window Geometry
        child_window_width = int(0.66*main_window_width)
        help_window_width  = int(0.33*main_window_width)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=child_window_width):

                ########################### 
                ###### Channel Block ######
                ###########################
                # Channel cleaning info
                channel_clean_list = list(self.options['allowed_clean_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'channel_clean'
                    dpg.add_text(f"{'Channel Cleaning Method':40}")
                    self.channel_clean_widget = dpg.add_combo(items=channel_clean_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)

                # Channel mapping info
                channel_list = list(self.options['allowed_channel_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'channel_list'
                    dpg.add_text(f"{'Channel Mapping':40}")
                    self.channel_list_widget = dpg.add_combo(items=channel_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)

                # Channel Montaging info
                montage_list = list(self.options['allowed_montage_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'montage'
                    dpg.add_text(f"{'Montage Method':40}")
                    self.montage_widget = dpg.add_combo(items=montage_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)

                ############################# 
                ###### Viability Block ######
                #############################
                dpg.add_separator()
                # Viability info
                viability_list = list(self.options['allowed_viability_args'].keys())
                with dpg.group(horizontal=True):
                    arg_var = 'viability'
                    dpg.add_text(f"{'Viability Method':40}")
                    self.montage_widget = dpg.add_combo(items=viability_list, callback=self.combo_callback, default_value=self.defaults[arg_var],width=int(0.5*child_window_width))
                    dpg.add_button(label="Help", callback=lambda sender, app_data: self.update_combo_help(self.channel_help,sender,app_data), tag=arg_var)


            # Text widget
            with dpg.group():
                dpg.add_text("Help:")
                self.channel_help = dpg.add_text("")