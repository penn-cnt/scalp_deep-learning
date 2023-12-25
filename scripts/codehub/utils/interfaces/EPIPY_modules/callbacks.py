import dearpygui.dearpygui as dpg

# Local imports to get documentation
from modules.addons.channel_clean import *
from modules.addons.channel_mapping import *
from modules.addons.channel_montage import *
from modules.addons.project_handler import *

class callback_handler:

    def __init__(self):
        pass

    def submit_fnc(self, sender, app_data):
        outputs = {}
        outputs['n_input']     = dpg.get_value(self.n_input_widget)
        outputs['n_offset']    = dpg.get_value(self.n_offset_widget)
        outputs['project']     = dpg.get_value(self.project_widget)
        outputs['multithread'] = dpg.get_value(self.multithread_widget)
        outputs['ncpu']        = dpg.get_value(self.ncpu_widget)
        outputs['t_start']     = dpg.get_value(self.t_start_widget)
        outputs['t_end']       = dpg.get_value(self.t_end_widget)
        for ikey,ival in outputs.items():
            print(f"{ikey}|{ival}")


    def combo_callback(self, sender, app_data):
        selected_item = dpg.get_value(sender)

    def radio_button_callback(self, sender, app_data):
        selected_item = dpg.get_value(sender)

    def folder_selection_callback(self, sender, app_data):
        selected_folder = dpg.get_value(sender)

    def update_help(self, help_obj, sender, app_data):
        button_alias = dpg.get_item_alias(sender)
        new_text     = f"{self.help[button_alias]}"
        new_text     = '\n'.join([new_text[i:i+self.nchar_help] for i in range(0, len(new_text), self.nchar_help)])
        dpg.set_value(help_obj, new_text)

    def update_combo_help(self, help_obj, sender, app_data):
        button_alias = dpg.get_item_alias(sender)
        if button_alias == 'project':
            combo_value = dpg.get_value(self.project_widget)
            new_text = getattr(project_handlers, combo_value.lower()).__doc__
        elif button_alias == 'channel_clean':
            combo_value = dpg.get_value(self.channel_clean_widget)
            new_text = getattr(channel_clean, f"clean_{combo_value.lower()}").__doc__
        elif button_alias == 'channel_list':
            combo_value = dpg.get_value(self.channel_list_widget)
            new_text = getattr(channel_mapping, f"mapping_{combo_value.lower()}").__doc__
        elif button_alias == 'montage':
            combo_value = dpg.get_value(self.montage_widget)
            new_text = getattr(channel_montage, f"montage_{combo_value.lower()}").__doc__

        new_text = new_text.replace("    ","")
        new_text = '\n'.join([new_text[i:i+self.nchar_help] for i in range(0, len(new_text), self.nchar_help)])
        dpg.set_value(help_obj, new_text)