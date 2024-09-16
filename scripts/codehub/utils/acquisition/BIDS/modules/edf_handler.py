import os 
import time
import getpass
from mne.io import read_raw_edf

# Local import
from modules.BIDS_handler import *
from modules.observer_handler import *
from modules.exception_handler import *
from utils.acquisition.BIDS.modules.data_backends import *

class edf_handler(Subject):

    def __init__(self,args):

        # Save the input objects
        self.args = args

        # Create the object pointers
        self.BH      = BIDS_handler()
        self.backend = return_backend(args.backend)

        # Get the data record
        self.get_data_record()

        # Create objects that interact with observers
        self.data_list     = []
        self.type_list     = []
        self.BIDS_keywords = {'root':self.args.bids_root,'datatype':None,'session':None,'subject':None,'run':None,'task':None}

    def workflow(self):
        """
        Run a workflow that downloads data from iEEG.org, creates the correct objects in memory, and saves it to BIDS format.
        """
        
        # Attach observers
        self.attach_objects()

        # Determine how to save the data
        self.get_inputs()

        # Begin downloading the data
        self.load_data_manager()

        # Save the data
        self.save_data()

        # Save the data record
        self.new_data_record = self.new_data_record.sort_values(by=['subject_number','session_number','run_number'])
        self.new_data_record.to_csv(self.data_record_path,index=False)

        # Remove if debugging
        if self.args.debug:
            os.system(f"rm -r {self.args.bids_root}*")

    def attach_objects(self):
        """
        Attach observers here so we can have each multiprocessor see the pointers correctly.
        """

        # Create the observer objects
        self._meta_observers = []
        self._data_observers = []

        # Attach observers
        self.add_meta_observer(BIDS_observer)
        self.add_data_observer(backend_observer)

    def get_inputs(self, multiflag=False, multiinds=None):
        """
        Create the input objects that track what files and times to download, and any relevant keywords for the BIDS process.
        For single core pulls, has more flexibility to set parameters. For multicore, we restrict it to a pre-built input_args.
        """

        # Check for an input csv to manually set entries
        if self.args.input_csv != None:
            
            # Read in the input data
            input_args = PD.read_csv(self.args.input_csv)

            # Pull out the relevant data pointers for required columns.
            self.edf_files = list(input_args['orig_filename'].values)

            # Get the unique identifier if provided
            if 'start' in input_args.columns:
                self.start_times=list(input_args['start'].values)
            else:
                self.start_times=[self.args.start for idx in range(input_args.shape[0])]

            # Get the unique identifier if provided
            if 'duration' in input_args.columns:
                self.durations=list(input_args['duration'].values)
            else:
                self.durations=[self.args.duration for idx in range(input_args.shape[0])]

            # Get the unique identifier if provided
            if 'uid' in input_args.columns:
                self.uid_list=list(input_args['uid'].values)
            else:
                self.uid_list=[self.args.uid for idx in range(input_args.shape[0])]

            # Get the subejct number if provided
            if 'subject_number' in input_args.columns:
                self.subject_list=list(input_args['subject_number'].values)
            else:
                self.subject_list=[self.args.subject_number for idx in range(input_args.shape[0])]

            # Get the session number if provided
            if 'session_number' in input_args.columns:
                self.session_list=list(input_args['session_number'].values)
            else:
                self.session_list=[self.args.session for idx in range(input_args.shape[0])]

            # Get the run number if provided
            if 'run_number' in input_args.columns:
                self.run_list=list(input_args['run_number'].values)
            else:
                self.run_list=[self.args.run for idx in range(input_args.shape[0])]

            # Get the task if provided
            if 'task' in input_args.columns:
                self.task_list=list(input_args['task'].values)

            # Get the target if provided
            if 'target' in input_args.columns:
                self.target_list = list(input_args['target'].values)
        else:
            # Get the required information if we don't have an input csv
            self.edf_files   = [self.args.dataset]
            self.start_times = [self.args.start]
            self.durations   = [self.args.duration]

            # Get the information that can be inferred
            if self.args.uid_number != None:
                self.uid_list = [self.args.uid_number]
            
            if self.args.subject_number != None:
                self.subject_list = [self.args.subject_number]
            
            if self.args.session != None:
                self.session_list = [self.args.session]
            
            if self.args.run != None:
                self.run_list = [self.args.run]
            
            if self.args.task != None:
                self.task_list = [self.args.task]

            if self.args.target != None:
                self.target_list = [self.args.target]

    def load_data_manager(self):
        """
        Loop over the ieeg file list and download data. If annotations, does a first pass to get annotation layers and times, then downloads.
        """

        # Set the reference variables we can use to avoid frequent checks of the data record
        self.record_checkfile = ''
        self.record_start     = -1
        self.record_duration  = -1

        # Loop over the requested data
        for idx in range(len(self.edf_files)):

            self.load_data(self.edf_files[idx])
                    
            # If successful, notify data observer. Else, add a skip
            if self.success_flag:
                self.notify_data_observers()
            else:
                self.data_list.append(None)

    def load_data(self,infile):
        raw           = read_raw_edf(infile,verbose=False)
        self.data     = raw.get_data().T
        self.channels = raw.ch_names
        self.fs       = raw.info.get('sfreq')

    def save_data(self):
        """
        Notify the BIDS code about data updates and save the results when possible.
        """
        
        # Loop over the data, assign keys, and save
        self.new_data_record = self.data_record.copy()
        for idx,iraw in enumerate(self.data_list):
            if iraw != None:

                # Update keywords
                self.keywords = {'filename':self.edf_files[idx],'root':self.args.bids_root,'datatype':self.type_list[idx],
                                 'session':self.session_list[idx],'subject':self.subject_list[idx],'run':self.run_list[idx],
                                 'task':'rest','fs':iraw.info["sfreq"]}
                self.notify_metadata_observers()

                # Save the data without events until a future release
                success_flag = self.BH.save_data_wo_events(iraw, debug=self.args.debug)

                # If the data wrote out correctly, update the data record
                if success_flag:
                    # Save the target info
                    try:
                        self.BH.save_targets(self.target_list[idx])
                    except:
                        pass

                    # Make the proposed data record row
                    self.current_record = PD.DataFrame([self.ieeg_files[idx]],columns=['orig_filename'])
                    self.current_record['source']         = 'edf_file'
                    self.current_record['creator']        = getpass.getuser()
                    self.current_record['gendate']        = time.strftime('%d-%m-%y', time.localtime())
                    self.current_record['uid']            = self.uid_list[idx]
                    self.current_record['subject_number'] = self.subject_list[idx]
                    self.current_record['session_number'] = self.session_list[idx]
                    self.current_record['run_number']     = self.run_list[idx]
                    self.current_record['start_sec']      = 1e-6*self.start_times[idx]
                    self.current_record['duration_sec']   = 1e-6*self.durations[idx]

                    # Add the datarow to the records
                    self.new_data_record = PD.concat((self.new_data_record,self.current_record))