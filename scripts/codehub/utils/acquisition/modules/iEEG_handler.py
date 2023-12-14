import numpy as np
import pandas as PD
from os import path
from tqdm import tqdm
from ieeg.auth import Session

# Multicore support
import threading
import multiprocessing
import concurrent.futures
from functools import partial

# Local imports
from modules.BIDS_handler import BIDS_handler

# Allows us to catch ieeg api errors
import ieeg.ieeg_api as IIA
from requests.exceptions import ReadTimeout as RTIMEOUT

# API timeout class
import signal
class TimeoutException(Exception):
    pass

class Timeout:
    def __init__(self, seconds=1, error_message='Function call timed out'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        print("A")
        raise TimeoutException(self.error_message)

    def __enter__(self):
        print("B")
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        print("C")
        signal.alarm(0)

class iEEG_download(BIDS_handler):

    def __init__(self, args, file_lock):
        
        # Store variables based on input params
        self.args           = args
        self.subject_path   = args.bidsroot+args.subject_file
        self.file_lock      = file_lock

        # Hard coded variables based on ieeg api
        self.n_retry        = 3
        self.global_timeout = 60
        self.clip_layer     = 'EEG clip times'
        self.natus_layer    = 'Imported Natus ENT annotations'

    def reset_variables(self):
            # Delete all variables in the object's namespace
            for var_name in list(self.__dict__.keys()):
                delattr(self, var_name)

    def get_annotations(self):

        # Get the clip times
        self.session_method_handler(0,1e6,annotation_flag=True)

        if self.success_flag:
            # Remove start clip time if it is just the machine starting up
            if self.clips[0].type.lower() == 'clip end' and self.clips[0].end_time_offset_usec == 2000:
                self.clips = self.clips[1:]

            # Manage edge cases
            if self.clips[0].type.lower() == 'clip end':
                self.clips = list(np.concatenate(([0],self.clips), axis=0))
            if self.clips[-1].type.lower() == 'clip start':
                self.clips = list(np.concatenate((self.clips,[self.end_time-self.start_time]), axis=0))

            clip_vals = []
            for iclip in self.clips:
                try:
                    clip_vals.append(iclip.start_time_offset_usec)
                except AttributeError:
                    clip_vals.append(iclip)

            # Turn the clip times into start and end arrays
            self.clip_start_times = np.array([iclip for iclip in clip_vals[::2]])
            self.clip_end_times   = np.array([iclip for iclip in clip_vals[1::2]])
            self.clip_durations   = self.clip_end_times-self.clip_start_times

            # Match the annotations to the clips
            self.annotations      = {ival:{} for ival in range(self.clip_start_times.size)}
            self.annotation_flats = []
            for annot in self.raw_annotations:
                time = annot.start_time_offset_usec
                desc = annot.description
                for idx, istart in enumerate(self.clip_start_times):
                    if (time >= istart) and (time <= self.clip_end_times[idx]):
                        event_time_shift = (time-istart)
                        self.annotations[idx][event_time_shift] = desc
                        self.annotation_flats.append(desc)

    def download_by_cli(self, uid, file, target, start, duration):

        # Store the ieeg filename
        self.uid          = uid
        self.current_file = file
        self.target       = target
        self.success_flag = False

        # Loop over clips
        if self.success_flag == True:
            BIDS_handler.__init__(self)
            self.session_method_handler(start,duration)
            if self.success_flag == True:
                BIDS_handler.get_channel_type(self)
                BIDS_handler.make_info(self)
                BIDS_handler.add_raw(self)

        # Save the bids files if we have any data
        try:
            if len(self.raws) > 0:
                BIDS_handler.event_mapper(self)
                BIDS_handler.save_bids(self)
        except AttributeError:
            pass

        # Clear namespace of variables for file looping
        BIDS_handler.reset_variables(self)
        self.reset_variables()

    def download_by_annotation(self, uid, file, target):

        # Store the ieeg filename
        self.uid          = uid
        self.current_file = file
        self.target       = target
        self.success_flag = False

        # Get the annotation times
        self.get_annotations()

        # Loop over clips
        if self.success_flag == True:
            BIDS_handler.__init__(self)
            for idx,istart in tqdm(enumerate(self.clip_start_times), desc="Downloading Clip Data", total=len(self.clip_start_times), leave=False, disable=self.args.multithread):
                self.session_method_handler(istart, self.clip_durations[idx])
                if self.success_flag == True:
                    BIDS_handler.get_channel_type(self)
                    BIDS_handler.make_info(self)
                    BIDS_handler.add_raw(self)

        # Save the bids files if we have any data
        try:
            if len(self.raws) > 0:
                BIDS_handler.event_mapper(self)
                if self.file_lock != None:
                    with self.file_lock:
                        BIDS_handler.save_bids(self)
                else:
                    BIDS_handler.save_bids(self)
        except AttributeError as e:
            print(e)

        # Clear namespace of variables for file looping
        BIDS_handler.reset_variables(self)
        self.reset_variables()

    def session_method_handler(self,start,duration,annotation_flag=False):
        """
        Wrapper to call ieeg. Due to ieeg errors, we want to make sure we can try to call it a few times before giving up.

        Args:
            start (float): Start time (referenced to data start) in microseconds to request data from
            duration (float): Duration in microseconds of data to request
            annotation_flag (bool, optional): Flag whether we just want annotation data or not. Defaults to False.
        """

        n_attempts = 0
        while True:
            with Timeout(self.global_timeout):
                try:
                    self.session_method(start,duration,annotation_flag)
                    self.success_flag = True
                    break
                except (IIA.IeegConnectionError,IIA.IeegServiceError,TimeoutException,RTIMEOUT,TypeError) as e:
                    if n_attempts<self.n_retry:
                        sleep(5)
                        n_attempts += 1
                    else:
                        self.success_flag = False
                        fp = open(self.args.bidsroot+self.args.failure_file,"a")
                        fp.write("%s,%f,%f,%s\n" %(self.current_file,start,duration,e))
                        fp.close()
                        break
        print("DONE")

    def session_method(self,start,duration,annotation_flag):
        """
        Call ieeg.org for data and return data or annotations.

        Args:
            start (float): Start time (referenced to data start) in microseconds to request data from
            duration (float): Duration in microseconds of data to request
            annotation_flag (bool, optional): Flag whether we just want annotation data or not. Defaults to False.

        Raises:
            IndexError: If there are multiple sampling frequencies, bids does not readily support this. Alerts user and stops.
        """

        with Session(self.args.username,self.args.password) as session:
            
            # Open dataset session
            dataset = session.open_dataset(self.current_file)
            
            # Logic gate for annotation call (faster, no time data needed) or get actual data
            if not annotation_flag:

                # Get the channel names and integer representations for data call
                self.channels = dataset.ch_labels
                channel_cntr  = list(range(len(self.channels)))

                # If duration is greater than 10 min, break up the call. Make array of start,duration with max 10 min each chunk
                time_cutoff = int(10*60*1e6)
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
                
                # Get the samping frequencies
                self.fs = [dataset.get_time_series_details(ichannel).sample_rate for ichannel in self.channels]

                # Data quality checks before saving
                if np.unique(self.fs).size == 1:
                    self.fs = self.fs[0]
                else:
                    raise IndexError("Too many unique values for sampling frequency.")
            else:
                self.clips           = dataset.get_annotations(self.clip_layer)
                self.raw_annotations = dataset.get_annotations(self.natus_layer)
                self.start_time      = dataset.start_time
                self.end_time        = dataset.end_time
            session.close()

class ieeg_handler:

    def __init__(self,args,map_data,input_files):
        self.args        = args
        self.map_data    = map_data
        self.input_files = input_files

        # Get list of files to skip that already exist locally
        subject_path = self.args.bidsroot+self.args.subject_file
        if path.exists(subject_path):
            self.processed_files = PD.read_csv(subject_path)['iEEG file'].values
        else:
            self.processed_files = []

    def single_pull(self):
        
        # Create a dummy lock to pass the multithread lock
        self.file_lock = None

        for file_idx in range(self.input_files.size):
            self.pull_data(file_idx)

    def multicore_pull(self):

        # Create a lock to synchronize file access
        self.file_lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.ncpu) as executor:
            executor.map(self.pull_data, range(self.input_files.size))

    def pull_data(self,file_idx):

        # Loop over files
        IEEG = iEEG_download(self.args,self.file_lock)
        ifile = self.input_files[file_idx]
        if ifile not in self.processed_files:
            print("Downloading %s. (%04d/%04d)" %(ifile,file_idx,self.input_files.size))
            iid    = self.map_data['uid'].values[file_idx]
            target = self.map_data['target'].values[file_idx]
            if self.args.annotations:
                IEEG.download_by_annotation(iid,ifile,target)
                IEEG = iEEG_download(self.args,self.file_lock)
            else:
                IEEG.download_by_cli(iid,ifile,target,self.args.start,self.args.duration)
        else:
            print("Skipping %s. (%04d/%04d)" %(ifile,file_idx,self.input_files.size))