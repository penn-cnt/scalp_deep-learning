import re
import os
import time
import getpass
import numpy as np
from time import sleep
from typing import List
import ieeg.ieeg_api as IIA
from ieeg.auth import Session
from requests.exceptions import ReadTimeout as RTIMEOUT

# Local import
from modules.BIDS_handler import *
from modules.observer_handler import *
from modules.exception_handler import *
from utils.acquisition.BIDS.modules.data_backends import *

class ieeg_handler(Subject):

    # Create the observer objects
    _meta_observers = []
    _data_observers = []

    def __init__(self,args):

        # Save the input objects
        self.args             = args

        # Create the object pointers
        self.BH      = BIDS_handler()
        self.backend = return_backend(args.backend)

        # Get the data record
        self.get_data_record()

        # Create objects that interact with observers
        self.data_list     = []
        self.BIDS_keywords = {'root':self.args.bids_root,'datatype':None,'session':None,'subject':None,'run':None,'task':None}

    def workflow(self):
        """
        Run a workflow that downloads data from iEEG.org, creates the correct objects in memory, and saves it to BIDS format.
        """
        
        # Attach observers
        self.add_meta_observer(BIDS_observer)
        self.add_data_observer(backend_observer)
        
        # Get the iEEG password
        self.get_password()

        # Determine what files to download and to where
        self.get_inputs()

        # Begin downloading the data
        self.download_data_manager(self.args.annotations)

        # Save the data
        self.save_data()

    def get_password(self):

        # Determine the method to get passwords. Not all systems can use a keyring easily.
        try:
            import keyring
            method = 'keyring'
        except ModuleNotFoundError:
            method = 'getpass'

        # Get the password from the user or the keyring. If needed, add to keyring.
        if method == 'keyring':
            self.password = keyring.get_password("eeg_bids_ieeg_pass", self.args.username)
            if self.password == None:
                self.password = getpass.getpass("Enter your password. (This will be stored to your keyring): ")                            
                keyring.set_password("eeg_bids_ieeg_pass", self.args.username, self.password)
        elif method == 'getpass':
            self.password = getpass.getpass("Enter your password: ")

    def get_data_record(self):
        
        # Get the proposed data record
        self.data_record_path = self.args.bids_root+self.args.data_record

        # Check if the file exists
        if os.path.exists(self.data_record_path):
            self.data_record = PD.read_csv(self.data_record_path)
        else:
            self.data_record = PD.DataFrame(columns=['orig_filename','source','creator','gendate','uid','subject_number','session_number','run_number','start_sec','duration_sec'])   

    def get_inputs(self):

        # Check for an input csv to manually set entries
        if self.args.input_csv != None:
            
            # Read in the data
            input_args = PD.read_csv(self.args.input_csv)

            # Raise some exceptions if we find data we can't work with
            if 'orig_filename' not in input_args.columns:
                raise Exception("Please provide 'orig_filename' in the input csv file.")
            elif 'orig_filename' in input_args.columns:
                if 'start' not in input_args.columns and not self.args.annotations:
                    raise Exception("A 'start' column is required in the input csv if not using the --annotations flag.")
                elif 'duration' not in input_args.columns and not self.args.annotations:
                    raise Exception("A 'duration' column is required in the input csv if not using the --annotations flag.")
            
            # Handle situations where the user requested annotations but also provided times
            if self.args.annotations:
                if 'start' in input_args.columns or 'duration' in input_args.columns:
                    userinput = ''
                    while userinput.lower() not in ['y','n']:
                        userinput = input("--annotations flag set to True, but start times and durations were provided in the input. Override these times with annotations clips? (Yy/Nn)")
                    if userinput.lower() == 'n':
                        print("Ignoring --annotation flag. Using user provided times.")
                        self.args.annotations = False
                    if userinput.lower() == 'y':
                        print("Ignoring user provided times in favor of annotation layer times.")
                        if 'start' in input_args.columns: input_args.drop(['start'],axis=1,inplace=True)
                        if 'duration' in input_args.columns: input_args.drop(['duration'],axis=1,inplace=True)

            # Pull out the relevant data pointers. This is a required input column, no check necessary.
            self.ieeg_files = list(input_args['orig_filename'].values)

            # Get candidate keywords for missing columns
            self.ieegfile_to_keys()

            # Get the unique identifier if provided
            if 'uid' in input_args.columns:
                self.uid_list=list(input_args['uid'].values)
            
            # Get the subejct number if provided
            if 'subject_number' in input_args.columns:
                self.subject_list=list(input_args['subject_number'].values)

            # Get the session number if provided
            if 'session_number' in input_args.columns:
                self.session_list=list(input_args['session_number'].values)

            # Get the run number if provided
            if 'run_number' in input_args.columns:
                self.run_list=list(input_args['run_number'].values)

            # Get the task if provided
            if 'task' in input_args.columns:
                self.task_list=list(input_args['task'].values)

        else:
            # Get the required information if we don't have an input csv
            self.ieeg_files  = [self.args.dataset]
            self.start_times = [self.args.start]
            self.durations   = [self.args.duration]

            # Infer input information from filename
            self.ieegfile_to_keys()

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
        
        # Add an object to store information via annotation downloads
        if self.args.annotations:
            self.annot_files      = []
            self.start_times      = []
            self.durations        = []
            self.annotation_uid   = []
            self.annotation_sub   = []
            self.annotation_ses   = []
            self.run_list         = []
            self.annotation_flats = []
            
    def annotation_cleanup(self,ifile,iuid,isub,ises):
        """
        Restructure annotation information to be used as new inputs.
        """

        # Remove start clip time if it is just the machine starting up
        if self.clips[0].type.lower() == 'clip end' and self.clips[0].end_time_offset_usec == 2000:
            self.clips = self.clips[1:]

        # Manage edge cases
        if self.clips[0].type.lower() == 'clip end':
            self.clips = list(np.concatenate(([0],self.clips), axis=0))
        if self.clips[-1].type.lower() == 'clip start':
            self.clips = list(np.concatenate((self.clips,[self.ieeg_end_time-self.ieeg_start_time]), axis=0))

        clip_vals = []
        for iclip in self.clips:
            try:
                clip_vals.append(iclip.start_time_offset_usec)
            except AttributeError:
                clip_vals.append(iclip)

        # Turn the clip times into start and end arrays
        clip_start_times = np.array([iclip for iclip in clip_vals[::2]])
        clip_end_times   = np.array([iclip for iclip in clip_vals[1::2]])
        clip_durations   = clip_end_times-clip_start_times

        # Match the annotations to the clips
        annotations      = {ival:{} for ival in range(clip_start_times.size)}
        annotation_flats = []
        for annot in self.raw_annotations:
            time = annot.start_time_offset_usec
            desc = annot.description
            for idx, istart in enumerate(clip_start_times):
                if (time >= istart) and (time <= clip_end_times[idx]):
                    event_time_shift = (time-istart)
                    annotations[idx][event_time_shift] = desc
                    annotation_flats.append(desc)
        
        # Update the instance wide values
        self.annot_files.extend([ifile for idx in range(len(clip_start_times))])
        self.annotation_uid.extend([iuid for idx in range(len(clip_start_times))])
        self.annotation_sub.extend([isub for idx in range(len(clip_start_times))])
        self.annotation_ses.extend([ises for idx in range(len(clip_start_times))])
        self.start_times.extend(clip_start_times)
        self.durations.extend(clip_durations)
        self.run_list.extend(np.arange(len(clip_start_times))+1)
        self.annotation_flats.extend(annotation_flats)

    def ieegfile_to_keys(self):
        """
        Use the iEEG.org filename to determine keywords.
        """

        # Extract possible keywords from the ieeg filename
        self.uid_list     = []
        self.subject_list = []
        self.session_list = []
        self.run_list     = []
        for ifile in self.ieeg_files:

            # Create a match object to search for relevant subject and session data
            match = re.search(r'\D+(\d+)_\D+(\d+)', ifile)

            # Get numerical portions of filename that correspond to subject and session
            if match:
                candidate_uid = int(match.group(1))
                candidate_sub = match.group(1)
                candidate_ses = match.group(2)
            else:
                candidate_uid = None
                candidate_sub = None
                candidate_ses = None

            # Look for this informaion in the records
            iDF = self.data_record.loc[self.data_record.orig_filename==ifile]

            # If the data already exists, get its previous information
            if iDF.shape[0] > 0:

                # Get the existing information
                candidate_uid = iDF.iloc[0].uid
                candidate_sub = str(iDF.iloc[0].subject_number)
                candidate_ses = str(iDF.iloc[0].session_number)

            # Create the subject and session lists
            self.uid_list.append(candidate_uid)
            self.subject_list.append(candidate_sub)
            self.session_list.append(candidate_ses)
            self.run_list.append(1)

    def download_data_manager(self,annotation_flag):

        # Loop over the requested data

        for idx in range(len(self.ieeg_files)):

            # Download the data
            if annotation_flag:
                self.download_data(self.ieeg_files[idx],0,0,annotation_flag)
                self.annotation_cleanup(self.ieeg_files[idx],self.uid_list[idx],self.subject_list[idx],self.session_list[idx])
            else:
                self.download_data(self.ieeg_files[idx],self.start_times[idx],self.durations[idx],annotation_flag)
                if self.success_flag:
                    self.notify_data_observers()
                else:
                    self.data_list.append(None)

        # If downloading by annotations, now loop over the clip level info and save
        if annotation_flag:
            # Update the object pointers for subject, session, etc. info
            self.ieeg_files   = self.annot_files
            self.uid_list     = self.annotation_uid
            self.subject_list = self.annotation_sub
            self.session_list = self.annotation_ses

            # Loop over the file list that is expanded by all the annotations
            for idx in range(len(self.ieeg_files)):

                # Download the data
                self.download_data(self.ieeg_files[idx],self.start_times[idx],self.durations[idx],False)
                if self.success_flag:
                    self.notify_data_observers()
                else:
                    self.data_list.append(None)

    def save_data(self):
        
        # Loop over the data, assign keys, and save
        for idx,iraw in enumerate(self.data_list):
            if iraw != None:

                # Update keywords
                self.keywords = {'root':self.args.bids_root,'datatype':'eeg','session':self.session_list[idx],'subject':self.subject_list[idx],'run':self.run_list[idx],'task':'rest'}
                self.notify_metadata_observers()

                # Save the data
                self.BH.save_data_wo_events(iraw, debug=self.args.debug)

                # Make the proposed data record row
                self.current_record = PD.DataFrame([self.ieeg_files[idx]],columns=['orig_filename'])
                self.current_record['source']         = 'ieeg.org'
                self.current_record['creator']        = getpass.getuser()
                self.current_record['gendate']        = time.strftime('%d-%m-%y', time.localtime())
                self.current_record['uid']            = self.uid_list[idx]
                self.current_record['subject_number'] = self.subject_list[idx]
                self.current_record['session_number'] = self.session_list[idx]
                self.current_record['run_number']     = self.run_list[idx]
                self.current_record['start_sec']      = self.start_times[idx]
                self.current_record['duration_sec']   = self.durations[idx]

                # Add the datarow to the records
                self.data_record = PD.concat((self.data_record,self.current_record))
                self.data_record.to_csv(self.data_record_path,index=False)
                
                # Remove if debugging
                if self.args.debug:
                    os.system(f"rm -r {self.args.bids_root}*")

    ###############################################
    ###### IEEG Connection related functions ######
    ###############################################

    def download_data(self,ieegfile,start,duration,annotation_flag,n_retry=5):

        # Attempt connection to iEEG.org up to the retry limit
        self.global_timeout = 60
        n_attempts          = 0
        self.success_flag   = False
        while True:
            with Timeout(self.global_timeout,False):
                try:
                    self.ieeg_session(ieegfile,start,duration,annotation_flag)
                    self.success_flag = True
                    break
                except (IIA.IeegConnectionError,IIA.IeegServiceError,TimeoutException,RTIMEOUT,TypeError) as e:
                    if n_attempts<n_retry:
                        sleep(5)
                        n_attempts += 1
                    else:
                        print(f"Connection Error: {e}")
                        break

    def ieeg_session(self,ieegfile,start,duration,annotation_flag):
        """
        Call ieeg.org for data and return data or annotations.

        Args:
            start (float): Start time (referenced to data start) in microseconds to request data from
            duration (float): Duration in microseconds of data to request
            annotation_flag (bool, optional): Flag whether we just want annotation data or not. Defaults to False.

        Raises:
            IndexError: If there are multiple sampling frequencies, bids does not readily support this. Alerts user and stops.
        """

        with Session(self.args.username,self.password) as session:
            
            # Open dataset session
            dataset = session.open_dataset(ieegfile)
            
            # Logic gate for annotation call (faster, no time data needed) or get actual data
            if not annotation_flag:

                # Status
                print(f"Downloading {ieegfile} starting at {1e-6*start:011.2f} seconds for {1e-6*duration:08.2f} seconds.")

                # Get the channel names and integer representations for data call
                self.channels = dataset.ch_labels
                channel_cntr  = list(range(len(self.channels)))

                # If duration is greater than 10 min, break up the call. Make array of start,duration with max 10 min each chunk
                twin_min    = 10
                time_cutoff = int(twin_min*60*1e6)
                end_time    = start+duration
                ival        = start
                chunks      = []
                while ival < end_time:
                    if ival+time_cutoff >= end_time:
                        chunks.append([ival,end_time-ival])
                    else:
                        chunks.append([ival,time_cutoff])
                    ival += time_cutoff

                # Call data and concatenate calls if greater than 10 min
                self.data   = []
                for ival in chunks:
                    self.data.append(dataset.get_data(ival[0],ival[1],channel_cntr))
                if len(self.data) > 1:
                    self.data = np.concatenate(self.data)
                else:
                    self.data = self.data[0]

                # Apply the voltage factors
                self.data = 1e-6*self.data

                # Get the channel labels
                self.channels = dataset.ch_labels

                # Get the samping frequencies
                self.fs = [dataset.get_time_series_details(ichannel).sample_rate for ichannel in self.channels]

                # Data quality checks before saving
                if np.unique(self.fs).size == 1:
                    self.fs = self.fs[0]
                else:
                    raise Exception("Too many unique values for sampling frequency.")
            else:
                self.clips           = dataset.get_annotations(self.args.time_layer)
                self.raw_annotations = dataset.get_annotations(self.args.annot_layer)
                self.ieeg_start_time = dataset.start_time
                self.ieeg_end_time   = dataset.end_time
            session.close()

