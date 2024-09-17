# EEG BIDS Creation

EEG Bids is a package designed to convert timeseries data into BIDS-compliant datasets. As the push for standardized datasets grows, harmonizing how we collect and store data has become increasingly important.

## Features

Currently, the package supports:

- Pulling data from iEEG.org
- Converting raw EDF files to BIDS format

We aim to make it easy to add new data pull methods by using an observer coding style, allowing new code to integrate with just a few lines. For more details, refer to the contribution section.

Additionally, the package generates various sidecar files used by other components of the CNT codehub for a range of tasks.

## Files

### `EEG_BIDS.py`
This is the user-interface portion of the code. You can access detailed usage instructions by running:
```bash
python EEG_BIDS.py --help
```

## Folders

### `modules`
This folder contains the backend code that powers EEG Bids, providing functionality to convert and handle timeseries data.

### `samples`
Includes sample CLI calls and input files to help you get started using the package.

## Installation

EEG_BIDS uses a number of specific packages, and it can be time consuming to build an environment just for the purposes of this script. We recommend starting with the directions for installing the cnt-codehub python environment found [here](https://github.com/penn-cnt/CNT-codehub/blob/main/README.md). You can then modify the cnt_codehub.yaml file as needed to match your needs.

## Usage Examples

For a few example use cases, see [here](https://github.com/penn-cnt/CNT-codehub/blob/main/scripts/codehub/utils/acquisition/BIDS/samples/sample_cmds.txt)

## Contributing

Placeholder for contributing guidelines.
