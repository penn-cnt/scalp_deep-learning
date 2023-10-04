# Import the classes
from .channel_mapping import *
from .dataframe_manager import *
from .channel_clean import *
from .channel_montage import *
from .output_manager import *
from .data_viability import *

class datatype_handlers:
    """
    Class devoted the specific pipeline used to load a data type. This is meant to provide a clean reproducable pipeline.

    New functions should follow all the data processing steps up to preprocessing and feature extraction that are relevant to their data type and data set.
    """

    def __init__(self):
        pass

    def edf_handler(self):
        """
        Run pipeline to load EDF data.
        """

        # Import data into memory
        data_loader.load_edf(self)

        # Clean the channel names
        channel_clean.__init__(self)

        # Get the correct channels for this merger
        channel_mapping.__init__(self,self.args.channel_list)

        # Create the dataframe for the object with the cleaned labels
        dataframe_manager.__init__(self)
        dataframe_manager.column_subsection(self,self.channel_map_out)

        # Perform next steps only if we have a viable dataset
        if self.dataframe.shape[0] == 0:
            print("Skipping %s.\n(This could be due to poorly selected start and end times.)" %(self.infile))
            pass
        else:
            # Put the data into a specific montage
            montage_data = channel_montage.__init__(self)
            dataframe_manager.montaged_dataframe(self,montage_data,self.montage_channels)

            # Update the output list
            output_manager.update_output_list(self,self.montaged_dataframe.values)