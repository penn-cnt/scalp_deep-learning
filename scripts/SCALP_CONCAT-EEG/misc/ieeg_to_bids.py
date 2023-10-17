import re
import sys
import mne
import glob
import mne_bids
import argparse
import numpy as np
import pandas as PD
from os import path
from time import sleep
from ieeg.auth import Session
from mne_bids import BIDSPath, write_raw_bids

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
        raise TimeoutException(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)

class BIDS_handler:

    def __init__(self):
        self.raws      = []
        self.data_info = {'iEEG_id':self.current_file}
        self.get_subject_number()

    def reset_variables(self):
            # Delete all variables in the object's namespace
            for var_name in list(self.__dict__.keys()):
                delattr(self, var_name)

    def get_subject_number(self):

        files            = glob.glob(self.args.bidsroot+'sub-*')
        if len(files) > 0:
            self.subject_num  = max([int(ifile.split('sub-')[-1]) for ifile in files])+1
        else:
            self.subject_num = 1

    def get_channel_type(self, threshold=15):

        # Define the expression that gets lead info
        regex = re.compile(r"(\D+)(\d+)")

        # Get the outputs of each channel
        channel_expressions = [regex.match(ichannel) for ichannel in self.channels]

        # Make the channel types
        self.channel_types = []
        for iexpression in channel_expressions:
            if iexpression == None:
                self.channel_types.append('misc')
            else:
                lead = iexpression.group(1)
                contact = int(iexpression.group(2))
                if lead.lower() in ["ecg", "ekg"]:
                    self.channel_types.append('ecg')
                elif lead.lower() in ['c', 'cz', 'cz', 'f', 'fp', 'fp', 'fz', 'fz', 'o', 'p', 'pz', 'pz', 't']:
                    self.channel_types.append('eeg')
                else:
                    self.channel_types.append(1)

        # Do some final clean ups based on number of leads
        lead_sum = 0
        for ival in self.channel_types:
            if isinstance(ival,int):lead_sum+=1
        if lead_sum > threshold:
            remaining_leads = 'ecog'
        else:
            remaining_leads = 'seeg'
        for idx,ival in enumerate(self.channel_types):
            if isinstance(ival,int):self.channel_types[idx] = remaining_leads
        self.channel_types = np.array(self.channel_types)

        # Make the dictionary for mne
        self.channel_types = PD.DataFrame(self.channel_types.reshape((-1,1)),index=self.channels,columns=["type"])

    def make_info(self):
        self.data_info = mne.create_info(ch_names=list(self.channels), sfreq=self.fs)

    def add_raw(self):
        self.raws.append(mne.io.RawArray(self.data.T, self.data_info))

    def event_mapper(self):

        keys = np.unique(self.annotation_flats)
        vals = np.arange(keys.size)
        self.event_mapping = dict(zip(keys,vals))

    def save_bids(self):

        # Loop over all the raw data, add annotations, save
        for idx, raw in enumerate(self.raws):
            
            # Set the channel types
            raw.set_channel_types(self.channel_types.type)

            # Make the events file and save the results
            for itime in list(self.annotations[idx].keys()):
                desc   = self.annotations[idx][itime]
                index  = (1e-6*itime)*self.fs
                events = np.array([[int(index),0,self.event_mapping[desc]]])
                try:
                    bids_path = mne_bids.BIDSPath(root=self.args.bidsroot, datatype='eeg', session=self.args.session, subject='%04d' %(self.subject_num), run=idx+1, task='task')
                    write_raw_bids(bids_path=bids_path, raw=raw, events_data=events,event_id=self.event_mapping, allow_preload=True, format='EDF',verbose=False)
                except FileExistsError:
                    pass

        # Save the subject file info
        iDF = PD.DataFrame([[self.current_file,self.subject_num]],columns=['iEEG file','subject_number'])

        if not path.exists(self.subject_path):
            subject_DF = iDF.copy()
        else:
            subject_DF = PD.read_csv(subject_path)
            subject_DF = PD.concat((subject_DF,iDF))
        subject_DF['subject_number'] = subject_DF['subject_number'].astype(str).str.zfill(4)
        subject_DF.to_csv(self.subject_path,index=False)


class iEEG_handler(BIDS_handler):

    def __init__(self,args):
        self.args           = args
        self.n_retry        = 5
        self.global_timeout = 120
        self.clip_layer     = 'EEG clip times'
        self.natus_layer    = 'Imported Natus ENT annotations'
        self.subject_path   = args.bidsroot+args.subject_file

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
        
    def download_by_annotation(self, file):

        # Store the ieeg filename
        self.current_file = file
        self.success_flag = False

        # Get the annotation times
        self.get_annotations()

        # Loop over clips
        if self.success_flag == True:
            BIDS_handler.__init__(self)
            for idx,istart in enumerate(self.clip_start_times):
                self.session_method_handler(istart, self.clip_durations[idx])
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

    def session_method(self,start,duration,annotation_flag):

        with Session(self.args.username,self.args.password) as session:
            dataset       = session.open_dataset(self.current_file)
            if not annotation_flag:
                self.channels = dataset.ch_labels
                channel_cntr  = list(range(len(self.channels)))
                self.data     = dataset.get_data(start,duration,channel_cntr)
                self.fs       = [dataset.get_time_series_details(ichannel).sample_rate for ichannel in self.channels]

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

    def session_method_handler(self,start,duration,annotation_flag=False):

        n_attempts = 0
        while True:
            with Timeout(self.global_timeout):
                try:
                    self.session_method(start,duration,annotation_flag)
                    self.success_flag = True
                    break
                except (IIA.IeegConnectionError,IIA.IeegServiceError,TimeoutException,RTIMEOUT,TypeError) as e:
                    if n_attempts<self.n_retry:
                        print("Possible iEEG error. Trying again momentarily.")
                        sleep(5)
                        n_attempts += 1
                    else:
                        self.success_flag = False
                        fp = open(self.args.bidsroot+self.args.failure_file,"a")
                        fp.write("%s,%f,%f,%s\n" %(self.current_file,start,duration,e))
                        fp.close()
                        break
            
if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")

    ieeg_group = parser.add_argument_group('iEEG connection options')
    ieeg_group.add_argument("--username", type=str, required=True, help="Username for iEEG.org.")
    ieeg_group.add_argument("--password", type=str, required=True, help="Password for iEEG.org.")
    ieeg_group.add_argument("--dataset", type=str, help="iEEG.org Dataset name")
    ieeg_group.add_argument("--start", type=float, help="Start time of clip")
    ieeg_group.add_argument("--duration", type=float, help="Duration of clip")
    ieeg_group.add_argument("--failure_file", default='./failed_ieeg_calls.csv', type=str, help="CSV containing failed iEEG calls.")

    bids_group = parser.add_argument_group('BIDS options')
    bids_group.add_argument("--bidsroot", type=str, required=True, help="Bids Root Directory.")
    bids_group.add_argument("--session", type=str, required=True, help="Session Keyword for BIDS.")

    other_group = parser.add_argument_group('Other options')
    other_group.add_argument("--annotation_file", type=str, help="File of iEEG datasets to download by annotation.")
    other_group.add_argument("--subject_file", type=str, default='subject_map.csv', help="File mapping subject id to ieeg file. (Defaults to bidroot+'subject_map.csv)")

    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--cli", action='store_true', default=False, help="Use start and duration from this CLI.")
    selection_group.add_argument("--csv", type=str, help="CSV file with filename, start time, and duration.")
    selection_group.add_argument("--annotations", action='store_true', default=False, help="File with filenames to obtained annotated data from.")
    args = parser.parse_args()

    # Selection criteria
    IEEG = iEEG_handler(args)
    if args.cli:
        data = 1
    elif args.csv:
        pass
    elif args.annotations:
        if args.annotation_file == None:
            IEEG.download_by_annotation(args.dataset)
        else:
            
            # Get list of files to skip that already exist locally
            subject_path = args.bidsroot+args.subject_file
            if path.exists(subject_path):
                processed_files = PD.read_csv(subject_path)['iEEG file']
            else:
                processed_files = []

            # Loop over files and process as needed
            input_files = PD.read_csv(args.annotation_file).values[:,0]
            for ifile in input_files:
                if ifile not in processed_files:
                    IEEG.download_by_annotation(ifile)
                    IEEG = iEEG_handler(args)


    