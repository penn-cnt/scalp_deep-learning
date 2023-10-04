CNT Research Repository Template
================
![version](https://img.shields.io/badge/version-0.2.1-blue)
![pip](https://img.shields.io/pypi/v/pip.svg)
![https://img.shields.io/pypi/pyversions/](https://img.shields.io/pypi/pyversions/4)

This code is designed to facilitate the merging of scalp EEG data collected by different sources. This is a temporary stand-in for DN3, which was originally specced to be a method for preparing data for deep learning tasks. At present DN3 is not ready for lab-wide implementation. This code aims to only prepare the data for deep learning ingested, and leaves the metadata for creating a model to the user.

# Prerequisites
In order to use this repository, you must have access to Python 3+. 

# Installation

The python environment required to run this code can be found in the following location. [Concatenation YAML](/core_libraries/python/scalp/envs/CNT_ENVIRON_SCALP_CONCAT.yml)

This file can be installed using the following call to conda:

> conda create --name <env> --file CNT_ENVIRON_SCALP_CONCAT.yml

where <env> is the name of the environment you wish to save this work under.

More information about creating conda environments can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

# Documentation
```
%run main.py --help
usage: main.py [-h] [--input {CSV,MANUAL,GLOB}] [--n_input N_INPUT] [--dtype {EDF}] [--t_start T_START] [--t_end T_END] [--t_window T_WINDOW] [--multithread] [--ncpu NCPU] [--channel_list {HUP1020,RAW}]
               [--montage {HUP1020,COMMON_AVERAGE}] [--viability {VIABLE_DATA,VIABLE_COLUMNS}] [--interp] [--n_interp N_INTERP] [--no_preprocess_flag] [--preprocess_file PREPROCESS_FILE] [--no_feature_flag]
               [--feature_file FEATURE_FILE] [--outdir OUTDIR]

Simplified data merging tool.

optional arguments:
  -h, --help            show this help message and exit

Data Merging Options:
  --input {CSV,MANUAL,GLOB}
                        Choose an option:
                        CSV            : Use a comma separated file of files to read in. (default)
                        MANUAL         : Manually enter filepaths.
                        GLOB           : Use Python glob to select all files that follow a user inputted pattern.
  --n_input N_INPUT     Limit number of files read in. Useful for testing.
  --dtype {EDF}         Choose an option:
                        EDF            : EDF file formats. (default)
  --t_start T_START     Time in seconds to start data collection.
  --t_end T_END         Time in seconds to end data collection. (-1 represents the end of the file.)
  --t_window T_WINDOW   List of window sizes, effectively setting multiple t_start and t_end for a single file.
  --multithread         Multithread flag.
  --ncpu NCPU           Number of CPUs to use if multithread.

Channel label Options:
  --channel_list {HUP1020,RAW}
                        Choose an option:
                        HUP1020        : Channels associated with a 10-20 montage performed at HUP.
                        RAW            : Use all possible channels. Warning, channels may not match across different datasets.

Montage Options:
  --montage {HUP1020,COMMON_AVERAGE}
                        Choose an option:
                        HUP1020        : Use a 10-20 montage.
                        COMMON_AVERAGE : Use a common average montage.

Data viability Options:
  --viability {VIABLE_DATA,VIABLE_COLUMNS}
                        Choose an option:
                        VIABLE_DATA    : Drop datasets that contain a NaN column. (default)
                        VIABLE_COLUMNS : Use the minimum cross section of columns across all datasets that contain no NaNs.
  --interp              Interpolate over NaN values of sequence length equal to n_interp.
  --n_interp N_INTERP   Number of contiguous NaN values that can be interpolated over should the interp option be used.

Preprocessing Options:
  --no_preprocess_flag  Do not run preprocessing on data.
  --preprocess_file PREPROCESS_FILE
                        Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.

Feature Extraction Options:
  --no_feature_flag     Do not run feature extraction on data.
  --feature_file FEATURE_FILE
                        Path to preprocessing YAML file. If not provided, code will walk user through generation of a pipeline.

Output Options:
  --outdir OUTDIR       Output directory.
```

# Major Features Remaining
- Associating target variables with the each subject

# License
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

# Contact Us
Any questions should be directed to the data science team. Contact information is provided below:

[Brian Prager](mailto:bjprager@seas.upenn.edu)

