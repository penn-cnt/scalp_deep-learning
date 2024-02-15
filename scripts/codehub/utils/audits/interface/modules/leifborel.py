import pandas as PD
import dearpygui.dearpygui as dpg

class leifborel_handler:

    def __init__(self):
        pass

    def showLeifBorel(self):

        # Search String
        str_width = 15
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Search for: ':{str_width}}")
            self.leifborel_search_text_widget = dpg.add_input_text(width=int(0.65*self.width))
            self.leifborel_search_widget      = dpg.add_button(label="Show Results", width=int(0.10*self.width), callback=lambda sender, app_data:self.search_fnc(
                                                                self.leifborel_search_text_widget, self.leifborel_text_id, self.leifborel_search_type_widget, 'leifborel', sender, app_data))
            self.leifborel_reset_widget       = dpg.add_button(label="Reset", width=int(0.10*self.width), callback=lambda sender, app_data:self.reset_fnc(self.leifborel_text_id, 'leifborel', sender, app_data))
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Search by: ':{str_width}}")
            self.leifborel_search_type_widget = dpg.add_radio_button(items=['path','md5'], horizontal=True, default_value='path')
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Fuzzy Match? ':{str_width}}")
            self.leifborel_fuzzysearch_widget = dpg.add_radio_button(items=[True,False], horizontal=True, default_value=False)
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Apply to: ':{str_width}}")
            self.apply_bsc   = dpg.add_button(label="BSC", width=int(0.10*self.width))
            self.apply_cnt1  = dpg.add_button(label="cnt1", width=int(0.10*self.width))
            self.apply_cntfs = dpg.add_button(label="cnt-fs", width=int(0.10*self.width))
        dpg.add_spacer(height=10)
        dpg.add_separator()
        with dpg.group(horizontal=True):
            dpg.add_text(f"{'Sort by: ':{str_width}}")
            self.apply_bsc   = dpg.add_button(label="path", width=int(0.10*self.width))
            self.apply_cnt1  = dpg.add_button(label="size", width=int(0.10*self.width))
            self.apply_cntfs = dpg.add_button(label="last-modified-date", width=int(0.10*self.width))
            self.apply_cntfs = dpg.add_button(label="owner", width=int(0.10*self.width))
        dpg.add_spacer(height=10)

        # Make a widget to help shrink the folder structure
        self.nfolder_shrink  = 0 #9

        # Add a multiline text input widget
        height = 0.85*self.height_fnc()
        self.leifborel_text_id = dpg.add_input_text(multiline=True, readonly=True, width=0.95*self.width,height=height)
        self.show_all_data("/Users/bjprager/Documents/GitHub/CNT-codehub/scripts/codehub/utils/audits/interface/modules/samples/sample.audit",self.leifborel_text_id,'leifborel')
        
