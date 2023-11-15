# Data Acquisition

We offer a few options for acquiring and ingesting data.

## iEEG.org to EDF/BIDS

[ieeg_to_bids.py](../../scripts/codehub/utils/acquisition/ieeg_to_bids.py) Will acquire data from the iEEG.org platform and download it into a BIDS compliant EDF file format.

A sample call to the script follows:
```
python scripts/codehub/utils/acquisition/ieeg_to_bids.py --username BJPrager --password ******** --bidsroot /mnt/leif/littlab/users/bjprager/DATA/IEEG/BIDS/ --annotations --session preimplant --annotation_file /mnt/leif/littlab/users/bjprager/DATA/IEEG/targets.csv --skip 1500
```

where
- username/password: Are your login credentials for iEEG.org
- bidsroot: Is the root directory where data will be created
- annotations: Tells the code to download data according to the annotation clip times
- session: Is the base BIDS session keyword to use (i.e. the string before any numeric counter)
- annotation_file: Is a csv that tells the code which datasets to download. It also allows you to assign a deidentified unique id to each patient and a target varaible
- skip: How many lines to skip from the annotation file. Useful if you need to rerun the download script. The script will not download duplicates, but if the iEEG API is having issues with specific files, it will allow you to skip the attempts to download the data once more.

For more information, please see the [README.md ](scripts/codehub/utils/acquisition/README.md) on how to use the script.

## Streaming Data

[stream_ssh.py](../../scripts/codehub/utils/acquisition/stream_ssh.py) Will stream data via a SSH tunnel into memory on your current machine.

For more information, please see the [README.md ](scripts/codehub/utils/acquisition/README.md) on how to use the script.
