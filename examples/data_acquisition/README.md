# Data Acquisition

We offer a few options for acquiring and ingesting data.

## iEEG.org to EDF/BIDS

[ieeg_to_bids.py](../../scripts/codehub/utils/acquisition/ieeg_to_bids.py) Will acquire data from the iEEG.org platform and download it into a BIDS compliant EDF file format.

For more information, please see the [README.md ](scripts/codehub/utils/acquisition/README.md) on how to use the script.

## Streaming Data

[stream_ssh.py](../../scripts/codehub/utils/acquisition/stream_ssh.py) Will stream data via a SSH tunnel into memory on your current machine.

For more information, please see the [README.md ](scripts/codehub/utils/acquisition/README.md) on how to use the script.
