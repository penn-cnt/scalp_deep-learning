import dearpygui.dearpygui as dpg

def showPipelines():
    with dpg.group(horizontal=True):
        with dpg.child_window(width=300):

            dpg.add_text('Cropping')
            dpg.add_text('Original Resolution:')
            dpg.add_text('Width:', tag='originalWidth')
            dpg.add_text('Height:', tag='originalHeight')
            dpg.add_text('Current Resolution:')
            dpg.add_text('Width:', tag='currentWidth')
            dpg.add_text('Height:', tag='currentHeight')
            dpg.add_text('New Resolution')
            dpg.add_separator();
            dpg.add_text('Start Width')
            dpg.add_input_int(tag='startY', width=-1)
            dpg.add_separator();
            dpg.add_text('Start Height')
            dpg.add_input_int(tag='startX', width=-1)
            dpg.add_separator();
            dpg.add_text('End Width')
            dpg.add_input_int(tag='endY', width=-1)
            dpg.add_separator();
            dpg.add_text('End Height')
            dpg.add_input_int(tag='endX', width=-1)
            dpg.add_separator();
            dpg.add_button(label='Reset', width=-1)
            dpg.add_button(label='Apply Changes', width=-1)