import os
import dearpygui.dearpygui as dpg

# Local imports
import pipeline_manager as PM

# Interface imports
from EPIPY_modules.theme import applyTheme
from EPIPY_modules.pipelines import showPipelines
from EPIPY_modules.callbacks import callback_handler
from EPIPY_modules.dataimport import dataimport_handler
from EPIPY_modules.configuration import configuration_handler

class Interface(callback_handler,configuration_handler,dataimport_handler):

    def __init__(self,args,metadata):
        self.args     = args
        self.help     = metadata[0]
        self.types    = metadata[1]
        self.defaults = metadata[2]
        self.options  = metadata[3]
        self.show()

    def show(self):
        dpg.create_context()
        dpg.create_viewport(title='EPIPY: Epilepsy Processing and Interpretation using PYthon', width=1280, height=720, min_height=600, min_width=900)

        with dpg.window(tag="Main"):
            applyTheme()
            self.showTabBar()
            pass
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("Main", True)
        dpg.start_dearpygui()
        dpg.destroy_context()
        pass

    def showTabBar(self):
        with dpg.tab_bar():
            self.showTabs()
        pass

    def showTabs(self):
        dpg.add_texture_registry(show=False, tag='textureRegistry')
        with dpg.tab(label='Configurations'):
            configuration_handler.showConfiguration(self)
            pass
        with dpg.tab(label='Data Preparation'):
            dataimport_handler.showDataImport(self)
            pass
        with dpg.tab(label='Acceptance Criteria'):
            #showThresholding(self.callbacks)
            pass
        with dpg.tab(label='Preprocessing Options'):
            #showContourExtraction(self.callbacks)
            pass
        with dpg.tab(label='Feature Extraction'):
            #showInterpolation(self.callbacks)
            pass
        #self.callbacks.imageProcessing.disableAllTags()
        pass

class App:
    def __init__(self,args,metadata):
        self.interface = Interface(args,metadata)
        pass


if __name__ == '__main__':
    
    # Get the pathing to the pipeline manager. Allows us to find the argument file
    pipeline_path = os.path.dirname(os.path.abspath(PM.__file__))+'/'

    # Get the arguments
    args, metadata = PM.argument_handler(argument_dir=pipeline_path,require_flag=False)
    args           = vars(args)

    # Run the dearpygui app
    app = App(args,metadata)