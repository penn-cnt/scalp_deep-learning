import pyperclip
import numpy as np
import dearpygui.dearpygui as dpg

# Local imports to get documentation
from modules.addons.channel_clean import *
from modules.addons.channel_mapping import *
from modules.addons.channel_montage import *
from modules.addons.project_handler import *

class callback_handler:

    def __init__(self):
        pass

    ############################
    ###### Helper Functions ####
    ############################
    def height_fnc(self):
        """
        Find a suitable height for the yaml multiline text object.
        Could use a better method for figuring out a decent height.
        """

        height     = dpg.get_viewport_client_height()
        open_space = 1-self.yaml_frac
        modifier   = np.amin([open_space,np.log10(height/self.height)])
        if height>=self.height:
            scale = (self.yaml_frac+modifier)
        else:
            scale = (self.yaml_frac-modifier) 
        return height*scale
    
    def yaml_example_url(self):
        """"
        Copy the examples url to the clipboard of the user.
        """
        pyperclip.copy(self.url)

    ################################
    ###### Resizing Functions ######
    ################################

    def update_submit_widget(self, sender, app_data):
        """
        Fix the height of the submission text widget if rescaled.
        """
        widget_height = self.height_fnc()
        dpg.configure_item(self.submit_widget_text, height=widget_height)

    def update_yaml_input_preprocess_widget(self, sender, app_data):
        widget_height = self.height_fnc()
        dpg.configure_item(self.yaml_input_preprocess_widget, height=widget_height)

    def update_yaml_input_features_widget(self, sender, app_data):
        widget_height = self.height_fnc()
        dpg.configure_item(self.yaml_input_features_widget, height=widget_height)


    ###############################
    #### File/Folder Selection ####
    ###############################
    
    def init_folder_selection(self,obj,sender,app_data):
        """
        Intialize the folder selection here. Need to send the object to populate with a path, and using a direct show_item doesn't allow this.
        """

        # Make a file and folder dialoge item
        self.current_path_obj = obj
        try:
            dpg.add_file_dialog(directory_selector=True, show=False, callback=self.path_selection_callback, tag="folder_dialog_id", height=400)
        except SystemError:
            pass
        dpg.show_item("folder_dialog_id")

    def init_file_selection(self,obj,sender,app_data):
        
        # Make a file and folder dialoge item
        self.current_path_obj = obj
        try:
            dpg.add_file_dialog(directory_selector=False, show=False, callback=self.path_selection_callback, tag="file_dialog_id", height=400)
            dpg.add_file_extension(".*",parent="file_dialog_id")
        except SystemError:
            pass
        dpg.show_item("file_dialog_id")

    def path_selection_callback(self, sender, app_data):
        """
        Select a file/folder and update the folder path field.
        """
        selected_path = list(app_data['selections'].values())[0]
        dpg.set_value(self.current_path_obj,selected_path)

    ############################
    ###### Misc Functions ######
    ############################    

    def display_example_preprocess(self):
        dpg.configure_item(self.yaml_input_preprocess_widget, default_value=self.preprocess_example)

    def clear_preprocess(self):
        dpg.configure_item(self.yaml_input_preprocess_widget, default_value='')

    def display_example_features(self):
        dpg.configure_item(self.yaml_input_features_widget, default_value=self.features_example)

    def clear_features(self):
        dpg.configure_item(self.yaml_input_features_widget, default_value='')

    def combo_callback(self, sender, app_data):
        selected_item = dpg.get_value(sender)

    def radio_button_callback(self, sender, app_data):
        selected_item = dpg.get_value(sender)

    def update_help(self, help_obj, sender, app_data):
        button_alias = dpg.get_item_alias(sender)
        new_text     = f"{self.help[button_alias]}"
        new_text     = '\n'.join([new_text[i:i+self.nchar_help] for i in range(0, len(new_text), self.nchar_help)])
        dpg.set_value(help_obj, new_text)

    def update_combo_help(self, help_obj, sender, app_data):
        button_alias = dpg.get_item_alias(sender)
        if button_alias == 'project':
            combo_value = dpg.get_value(self.project_widget)
            try:
                new_text = getattr(project_handlers, combo_value.lower()).__doc__
            except AttributeError:
                new_text = self.options['allowed_project_args'][combo_value]
        elif button_alias == 'input':
            combo_value = dpg.get_value(self.input_widget)
            new_text    = self.options['allowed_input_args'][combo_value]
        elif button_alias == 'channel_clean':
            combo_value = dpg.get_value(self.channel_clean_widget)
            try:
                new_text = getattr(channel_clean, f"clean_{combo_value.lower()}").__doc__
            except AttributeError:
                new_text = self.options['allowed_clean_args'][combo_value]
        elif button_alias == 'channel_list':
            combo_value = dpg.get_value(self.channel_list_widget)
            try:
                new_text = getattr(channel_mapping, f"mapping_{combo_value.lower()}").__doc__
            except AttributeError:
                new_text = self.options['allowed_channel_args'][combo_value]
        elif button_alias == 'montage':
            combo_value = dpg.get_value(self.montage_widget)
            try:
                new_text = getattr(channel_montage, f"montage_{combo_value.lower()}").__doc__
            except AttributeError:
                new_text = self.options['allowed_montage_args'][combo_value]
        elif button_alias == 'viability':
            combo_value = dpg.get_value(self.viability_widget)
            new_text    = self.options['allowed_viability_args'][combo_value]

        # Remove tabs
        new_text = new_text.replace("    ","")
        
        # Set the text
        dpg.set_value(help_obj, new_text)

    def create_submit_cmd(self):

        """
        python pipeline_manager.py --input GLOB --glob_str "/mnt/leif/littlab/users/bjprager/DATA/IEEG/BIDS/*/*/sub*/*/eeg/*edf" 
        --preprocess_file /mnt/leif/littlab/users/bjprager/GitHub/scalp_deep-learning/user_data/derivative/preprocessing_grid/configs/preprocessing.yaml 
        --feature_file /mnt/leif/littlab/users/bjprager/GitHub/scalp_deep-learning/user_data/derivative/preprocessing_grid/configs/features.yaml 
        --outdir /mnt/leif/littlab/users/bjprager/GitHub/scalp_deep-learning/user_data/derivative/preprocessing_grid/output/ 
        --targets --n_input 500 --t_window 5 --t_overlap 0.4 --ncpu 32 --multithread 
        --exclude /mnt/leif/littlab/users/bjprager/GitHub/scalp_deep-learning/user_data/derivative/preprocessing_grid/output/excluded.txt
        """

        # Base command to the code
        base_cmd = 'python pipeline_manager.py'
        excl     = ['',None]
        
        while True:
            # Input handler
            input_type = dpg.get_value(self.input_widget)
            base_cmd   = f"{base_cmd} --input {input_type}"
            input_path = dpg.get_value(self.input_path_widget)
            if input_path not in excl:
                base_cmd   = f"{base_cmd} --input_str {input_path}"
            
            # Output Handler
            output_path = dpg.get_value(self.output_widget_text)
            if output_path in excl:
                base_cmd = "Please provide an output path."
                break
            else:
                base_cmd = f"{base_cmd} --outdir {output_path}"

            # Multiprocessing handler
            multithread = dpg.get_value(self.multithread_widget)
            ncpu        = dpg.get_value(self.ncpu_widget)
            if multithread:
                base_cmd = f"{base_cmd} --multithread --ncpu {ncpu}"

            # File jumping options
            n_input  = dpg.get_value(self.n_input_widget)
            n_offset = dpg.get_value(self.n_offset_widget)
            if n_input not in excl and n_input > 0 :
                base_cmd = f"{base_cmd} --n_input {n_input}"
            if n_offset not in excl:
                base_cmd = f"{base_cmd} --n_offset {n_offset}"

            # Timing handler
            t_start   = dpg.get_value(self.t_start_widget)
            t_end     = dpg.get_value(self.t_end_widget)
            t_window  = dpg.get_value(self.t_window_widget)
            t_overlap = np.round(dpg.get_value(self.t_overlap_widget),2)
            if t_start not in excl and t_start > 0:
                base_cmd = f"{base_cmd} --t_start {t_start}"
            if t_end not in excl and t_end != -1:
                base_cmd = f"{base_cmd} --t_start {t_end}"
            if t_window not in excl:
                base_cmd = f"{base_cmd} --t_window {t_window}"
            if t_overlap not in excl and t_overlap > 0:
                base_cmd = f"{base_cmd} --t_overlap {t_overlap}"

            # Channel options
            channel_clean   = dpg.get_value(self.channel_clean_widget)
            channel_list    = dpg.get_value(self.channel_list_widget)
            channel_montage = dpg.get_value(self.montage_widget)
            base_cmd        = f"{base_cmd} --channel_clean {channel_clean} --channel_list {channel_list} --montage {channel_montage}"

            # Data Viability options
            viability = dpg.get_value(self.viability_widget) 
            interp    = dpg.get_value(self.interp_widget)
            n_interp  = dpg.get_value(self.n_interp_widget)
            base_cmd  = f"{base_cmd} --viability {viability}"
            if interp:
                base_cmd  = f"{base_cmd} --interp --n_interp {n_interp}"

            # Verbose
            silent = dpg.get_value(self.verbose_widget)
            if silent:
                base_cmd = f"{base_cmd} --silent"

            break

        # Make a display version and update
        self.submit_str      = base_cmd
        self.submit_str_disp = '\n'.join([self.submit_str[i:i+self.nchar_child] for i in range(0, len(self.submit_str), self.nchar_child)])
        dpg.set_value(self.submit_widget_text, self.submit_str_disp)
