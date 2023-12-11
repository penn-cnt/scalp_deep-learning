import dearpygui.dearpygui as dpg

def folder_selection_callback(sender, app_data):
    selected_folder = dpg.get_value(sender)

def showConfiguration(main_window_width = 1280):

    # Child Window Geometry
    child_window_width = int(0.66*main_window_width)

    with dpg.file_dialog(directory_selector=True, show=False, callback=folder_selection_callback, height=400) as file_dialog:
        dpg.add_input_text(label="Output Folder", width=400, callback=folder_selection_callback)
        dpg.add_button(label="Select Folder", callback=lambda: dpg.configure_item(file_dialog, show=True))

    with dpg.group(horizontal=True):
        with dpg.child_window(width=960):

            # Output directory selection
            with dpg.group(horizontal=True):
                dpg.add_text(f"{'Output Directory':20}")
                dpg.add_input_text(width=int(0.5*child_window_width))
                dpg.add_button(label="Select Folder", callback=lambda: dpg.configure_item(file_dialog, show=True))
            
            # Multithread Options
            dpg.add_text(f"{'Multithreaded':20}")
            with dpg.group(horizontal=True):
                ncpu_widget = dpg.add_text(f"{'# CPUs':20}", tag='nCPU')
                dpg.add_input_int()

                
