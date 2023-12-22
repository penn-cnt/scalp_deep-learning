from modules.addons.data_loader import *

if __name__ == '__main__':
    fname='/mnt/leif/littlab/cache/Human_Data/BIDS/sub-00013/ses-preimplant001/eeg/sub-00013_ses-preimplant001_task-task_run-10_eeg.edf'
    DL=data_loader()
    data=DL.direct_inputs(fname,'ssh_edf',ssh_host='borel.seas.upenn.edu',ssh_username='bjprager')
    print(data)