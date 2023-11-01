from sys import exit

# Import the classes
from .data_loader import *
from .metadata_handler import *
from .target_loader import *
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .output_manager import *
from .data_viability import *

class project_handlers:
    """
    Class devoted the specific pipeline used to load data according to project needs. This is meant to provide a clean reproducable pipeline.

    New functions should follow all the data processing steps up to preprocessing and feature extraction that are relevant to their data type and data set.
    """

    def __init__(self):
        pass

    def scalp_00(self):
        """
        Run pipeline to load EDF data for a scalp project.
        """

        # Import data into memory
        load_flag = data_loader.pipeline(self,'edf') 

        if load_flag:
            # Clean the channel names
            channel_clean.pipeline(self)

            # Get the correct channels for this merger
            channel_mapping.__init__(self,self.args.channel_list)

            # Create the dataframe for the object with the cleaned labels
            dataframe_manager.__init__(self)
            dataframe_manager.column_subsection(self,self.channel_map_out)

            # Perform next steps only if we have a viable dataset
            if self.dataframe.shape[0] <= int(max(self.metadata[self.file_cntr]['fs'])):
                pass
            else:
                # Put the data into a specific montage
                montage_data = channel_montage.pipeline(self)
                dataframe_manager.montaged_dataframe(self,montage_data,self.montage_channels)

                # Update the output list
                output_manager.update_output_list(self,self.montaged_dataframe.values)