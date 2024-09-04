from mne_bids import BIDSPath,write_raw_bids

# Local Imports
from modules.observer_handler import *

class BIDS_observer(Observer):

    def listen_metadata(self):

        # Define the required BIDS keywords
        BIDS_keys = ['root','datatype','session','subject','run','task']

        # Populate the bids dictionary with the new values
        for ikey,ivalue in self.keywords.items():
            if ikey in BIDS_keys:
                self.BIDS_keywords[ikey]=ivalue

        # If all keywords are set, send the new pathing to the BIDS handler.
        if all(self.BIDS_keywords.values()):
            self.BH.update_path(self.BIDS_keywords)

class BIDS_handler:

    def __init__(self):
        pass

    def update_path(self,keywords):
        """
        Update the bidspath.
        """

        self.current_keywords = keywords
        self.bids_path = BIDSPath(root=str(keywords['root']), 
                                  datatype=str(keywords['datatype']), 
                                  session=str(keywords['session']), 
                                  subject=str(keywords['subject']),
                                  run=int(keywords['run']), 
                                  task=str(keywords['task']))
        
    def create_events(self):

        # Make the events file and save the results
        events  = []
        alldesc = []
        for iannot in self.annotations[idx].keys():
            desc  = self.annotations[idx][iannot]
            index = (1e-6*iannot)*self.fs
            events.append([index,0,self.event_mapping[desc]])
            alldesc.append(desc)
        events = np.array(events)

    def save_data_w_events(self, raw, events, event_mapping, debug=False):

        # Save the bids data
        write_raw_bids(bids_path=self.bids_path, raw=raw, events_data=events,event_id=event_mapping, allow_preload=True, format='EDF',verbose=False)

    def save_data_wo_events(self, raw, debug=False):

        # Save the bids data
        try:
            write_raw_bids(bids_path=self.bids_path, raw=raw, allow_preload=True, format='EDF',verbose=False)
        except Exception as e:
            if debug:
                print(f"Write error: {e}")
