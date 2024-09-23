# API timeout class
import signal
class TimeoutException(Exception):
    pass

class Timeout:
    """
    Manage timeouts to the iEEG.org API call. It can go stale, and sit for long periods of time otherwise.
    """

    def __init__(self, seconds=1, multiflag=False, error_message='Function call timed out'):
        self.seconds       = seconds
        self.error_message = error_message
        self.multiflag     = multiflag

    def handle_timeout(self, signum, frame):
        raise TimeoutException(self.error_message)

    def __enter__(self):
        if not self.multiflag:
            signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.seconds)
        else:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.multiflag:
            signal.alarm(0)
        else:
            pass

class DataExists:
    """
    Checks data records for existing data.
    """

    def __init__(self,data_record):
        self.data_record      = data_record
        self.record_checkfile = ''
        self.record_start     = -1
        self.record_duration  = -1

    def check_default_records(self,checkfile,checkstart,checkduration):
        """
        Check the data record for data that matched the current query.

        Args:
            checkfile (_type_): Current ieeg.org filename.
            checkstart (_type_): Current iEEG.org start time in seconds.
            checkduration (_type_): Current iEEG.org duration in seconds.

        Returns:
            bool: True if no data found in record. False is found.
        """

        # Update file mask as needed
        if checkfile != self.record_checkfile:
            self.record_checkfile = checkfile
            self.record_file_mask = (self.data_record['orig_filename'].values==checkfile)
        if checkstart != self.record_start:
            self.record_start      = checkstart
            self.record_start_mask = (self.data_record['start_sec'].values==checkstart)
        if checkduration != self.record_duration:
            self.record_duration      = checkduration
            self.record_duration_mask = (self.data_record['duration_sec'].values==checkduration)

        # Get the combined mask
        mask = self.record_file_mask*self.record_start_mask*self.record_duration_mask

        # Check for any existing records
        return not(any(mask))
