# EEG BIDS Creation

This code is designed to place timeseries data into BIDS format. At present, it can take inputs from iEEG.org or local edf data. Future releases will expand the supported input formats. Data just needs to be sent to the  backend_handler, which gets converted to an MNE raw object, to allow for more data types.

## Examples
For a few example use cases, see [here](https://github.com/penn-cnt/CNT-codehub/blob/main/scripts/codehub/utils/acquisition/BIDS/samples/sample_cmds.txt)