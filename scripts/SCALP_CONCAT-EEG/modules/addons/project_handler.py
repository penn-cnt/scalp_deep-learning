from sys import exit

# Import the add on classes
from modules.addons.data_loader import *
from modules.addons.channel_clean import *
from modules.addons.channel_mapping import *
from modules.addons.channel_montage import *
from modules.addons.preprocessing import *
from modules.addons.features import *

# Import the core classes
from modules.core.metadata_handler import *
from modules.core.target_loader import *
from modules.core.dataframe_manager import *
from modules.core.output_manager import *
from modules.core.data_viability import *

class project_handlers:
    """
    Class devoted the specific pipeline used to load data according to project needs. This is meant to provide a clean reproducable pipeline.

    New functions should follow all the data processing steps up to preprocessing and feature extraction that are relevant to their data type and data set.
    """

    def __init__(self):
        pass

    def file_manager(self):
        """
        Loop over the input files and send them to the correct data handler.

        Args:
            infiles (str list): Path to each dataset
            start_times (float list): Start times in seconds to start sampling
            end_times (float list): End times in seconds to end sampling
        """

        # Intialize a variable that stores the previous filepath. This allows us to cache data and only read in as needed. (i.e. new path != old path)
        self.oldfile = None  

        # Loop over files to read and store each ones data
        nfile = len(self.infiles)
        desc  = "Initial load with id %s:" %(self.unique_id)
        for ii,ifile in tqdm(enumerate(self.infiles), desc=desc, total=nfile, bar_format=self.bar_frmt, position=self.worker_number, leave=False, disable=self.args.silent):            
        
            # Save current file info
            self.infile    = ifile
            self.t_start   = self.start_times[ii]
            self.t_end     = self.end_times[ii]
            
            # Initialize the metadata container
            self.file_cntr = ii

            # Apply project specific pipeline
            self.project_logic()
            
            # Update file strings for cached read in
            self.oldfile = self.infile

    def preprocessing_manager(self):

        # Apply preprocessing as needed
        if not self.args.no_preprocess_flag:
            
            # Barrier the code for better output formatting
            if self.args.multithread:
                self.barrier.wait()

                # Add a wait for proper progress bars
                time.sleep(self.worker_number)

                # Clean up the screen
                if self.worker_number == 0:
                    sys.stdout.write("\033[H")
                    sys.stdout.flush()

            # Process
            preprocessing.__init__(self)

    def feature_manager(self):

        if not self.args.no_feature_flag:
            if self.args.multithread:
                self.barrier.wait()

                # Add a wait for proper progress bars
                time.sleep(self.worker_number)

                # Clean up the screen
                if self.worker_number == 0:
                    sys.stdout.write("\033[H")
                    sys.stdout.flush()
            features.__init__(self)

    def target_manager(self):

        if self.args.targets:
            for ikey in self.metadata.keys():
                ifile   = self.metadata[ikey]['file']
                target_loader.load_targets(self,ifile,'bids','target')

    def project_logic(self):

        # Case statement the workflow
        if self.args.project.lower() == 'scalp_00':
            project_handlers.scalp_00(self)

    ##########################
    #### Template Project ####
    ##########################

    def template(self):

        # Import data into memory
        load_flag = data_loader.pipeline(self,'edf')      # Load flag is a boolean that lets us know if the current data loaded correctly

        # If data loaded, begin the processing portion
        if load_flag:
            # Clean the channel names
            channel_clean.pipeline(self)

            # Get the correct channels for this merger
            channel_mapping.pipeline(self)

            # Once we have the cleaned channel names, and the appropriate column slices, make a dataframe
            # Data up to this point is kept as a raw array due to variable input data formats and because
            # dataframes take up more memory and have slower operations.
            # This can go anywhere after the initial data load, and before the preprocessing, montaging, and feature extraction.
            dataframe_manager.__init__(self)
            dataframe_manager.column_subsection(self,self.channel_map_out)            


    ###################################
    #### User Provided Logic Below ####
    ###################################

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
            channel_mapping.pipeline(self)

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