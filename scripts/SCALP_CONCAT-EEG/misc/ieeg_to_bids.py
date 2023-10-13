import re
import sys
import mne
import glob
import mne_bids
import argparse
import numpy as np
import pandas as PD
from time import sleep
from ieeg.auth import Session
from mne_bids import BIDSPath, write_raw_bids
from ieeg.ieeg_api import IeegConnectionError

class BIDS_handler:

    def __init__(self):
        self.raws      = []
        self.data_info = {'iEEG_id':self.current_file}
        self.get_subject_number()

    def get_subject_number(self):

        files            = glob.glob(self.args.bidsroot+'sub*')
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
        self.data_info = mne.create_info(ch_names=list(self.channels), sfreq=self.fs) #, ch_types=self.channel_types)

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
                index  = (1e-6*itime)/self.fs
                events = np.array([[int(index),0,self.event_mapping[desc]]])
                try:
                    bids_path = mne_bids.BIDSPath(root=self.args.bidsroot, datatype='eeg', session=self.args.session, subject='%04d' %(self.subject_num), run=idx+1, task='task')
                    write_raw_bids(bids_path=bids_path, raw=raw, events_data=events,event_id=self.event_mapping, allow_preload=True, format='EDF',verbose=False)
                except FileExistsError:
                    pass


class iEEG_handler(BIDS_handler):

    def __init__(self,args):
        self.args        = args
        self.n_retry     = 5
        self.clip_layer  = 'EEG clip times'
        self.natus_layer = 'Imported Natus ENT annotations'

    def get_annotations(self,file):

        # Get the clip times
        with Session(self.args.username,self.args.password) as session:
            dataset     = session.open_dataset(self.current_file)
            clips       = dataset.get_annotations(self.clip_layer)
            annotations = dataset.get_annotations(self.natus_layer)
            end_time = dataset.end_time
        session.close()

        # Remove start clip time if it is just the machine starting up
        if clips[0].type.lower() == 'clip end' and clips[0].end_time_offset_usec == 2000:
            clips = clips[1:]

        # Manage edge cases
        if clips[0].type.lower() == 'clip end':
            clips = list(np.concatenate(([0],clips), axis=0))
        if clips[-1].type.lower() == 'clip start':
            clips = list(np.concatenate((clips,[end_time]), axis=0))

        clip_vals = []
        for iclip in clips:
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
        for annot in annotations:
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

        # Get the annotation times
        self.get_annotations(file)

        # Loop over clips
        BIDS_handler.__init__(self)
        for idx,istart in enumerate(self.clip_start_times):
            self.session_method_handler(file, istart, self.clip_durations[idx])
            BIDS_handler.get_channel_type(self)
            BIDS_handler.make_info(self)
            BIDS_handler.add_raw(self)
        BIDS_handler.event_mapper(self)
        BIDS_handler.save_bids(self)

    def session_method(self,start,duration):

        with Session(self.args.username,self.args.password) as session:
            dataset       = session.open_dataset(self.current_file)
            self.channels = dataset.ch_labels
            channel_cntr  = list(range(len(self.channels)))
            self.data     = dataset.get_data(start,duration,channel_cntr)
            self.fs       = [dataset.get_time_series_details(ichannel).sample_rate for ichannel in self.channels]
            session.close()

        # Data quality checks before saving
        if np.unique(self.fs).size == 1:
            self.fs = self.fs[0]
        else:
            raise IndexError("Too many unique values for sampling frequency.")

    def session_method_handler(self,file,start,duration):

        n_attempts = 0
        while True:
            try:
                self.session_method(start,duration)
                break
            except IeegConnectionError as e:
                if n_attempts<self.n_retry:
                    print("Possible iEEG error. Trying again momentarily.")
                    sleep(5)
                    n_attempts += 1
                else:
                    fp = open(self.args.failure_file,"a")
                    fp.write("%s,%f,%f\n" %(file,start,duration))
                    fp.close()
                    break

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")

    ieeg_group = parser.add_argument_group('iEEG connection options')
    ieeg_group.add_argument("--username", type=str, required=True, help="Username for iEEG.org.")
    ieeg_group.add_argument("--password", type=str, required=True, help="Password for iEEG.org.")
    ieeg_group.add_argument("--dataset", type=str, required=True, help="iEEG.org Dataset name")
    ieeg_group.add_argument("--start", type=float, help="Start time of clip")
    ieeg_group.add_argument("--duration", type=float, help="Duration of clip")
    ieeg_group.add_argument("--failure_file", default='./failed_ieeg_calls.csv', type=str, help="CSV containing failed iEEG calls.")

    bids_group = parser.add_argument_group('BIDS options')
    bids_group.add_argument("--bidsroot", type=str, required=True, help="Bids Root Directory.")
    bids_group.add_argument("--session", type=str, required=True, help="Session Keyword for BIDS.")

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
        IEEG.download_by_annotation(args.dataset)
    