import dearpygui.dearpygui as dpg

def submit_fnc(sender, app_data):
    pass

def help_fnc(sender,app_data):
    pass

def combo_callback(sender,app_data):
    selected_item = dpg.get_value(sender)

def radio_button_callback(sender, app_data):
    selected_item = dpg.get_value(sender)

def folder_selection_callback(sender, app_data):
    selected_folder = dpg.get_value(sender)

def showConfiguration(main_window_width = 1280):

    # Child Window Geometry
    child_window_width = int(0.66*main_window_width)

    with dpg.file_dialog(directory_selector=True, show=False, callback=folder_selection_callback, height=400) as file_dialog:
        dpg.add_input_text(label="Output Folder", width=400, callback=folder_selection_callback)
        dpg.add_button(label="Select Folder", callback=lambda: dpg.configure_item(file_dialog, show=True))

    with dpg.group(horizontal=True):
        with dpg.child_window(width=child_window_width):

            # Output directory selection
            with dpg.group(horizontal=True):
                dpg.add_text(f"{'Output Directory':20}")
                output_widget_text   = dpg.add_input_text(width=int(0.5*child_window_width))
                output_widget_button = dpg.add_button(label="Select Folder", callback=lambda: dpg.configure_item(file_dialog, show=True))
                dpg.add_button(label="Help", callback=help_fnc)

            # Multithread Options
            with dpg.group(horizontal=True):
                dpg.add_text(f"{'Multithreaded':20}")
                radio_multithread_true  = dpg.add_radio_button(items=[True,False], callback=radio_button_callback, horizontal=True, default_value=False)
                dpg.add_button(label="Help", callback=help_fnc)
            with dpg.group(horizontal=True):
                dpg.add_text(f"{'# CPUs':20}", tag='nCPU')
                dpg.add_input_int(default_value=1,step_fast=4)
                dpg.add_button(label="Help", callback=help_fnc)

            # Input options for skipping around files
            with dpg.group(horizontal=True):
                dpg.add_text(f"{'# of input files':20}")
                dpg.add_input_int(default_value=0,step_fast=25)
                dpg.add_button(label="Help", callback=help_fnc)
            with dpg.group(horizontal=True):
                dpg.add_text(f"{'# of skipped inputs':20}")
                dpg.add_input_int(default_value=0,step_fast=25)
                dpg.add_button(label="Help", callback=help_fnc)

            # Setup the project selections
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(f"{'Project Workflow':20}")
                combo_box = dpg.add_combo(items=["Scalp 00"], callback=combo_callback)
                dpg.add_button(label="Help", callback=help_fnc)

            # Submit a job
            dpg.add_spacing(height=400)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Submit Job", callback=submit_fnc)