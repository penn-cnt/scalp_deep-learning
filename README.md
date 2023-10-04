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
function test() {
  console.log("notice the blank line before this function?");
}
```

# Major Features Remaining
- Associating target variables with the each subject

# License
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

# Contact Us
Any questions should be directed to the data science team. Contact information is provided below:

[Brian Prager](mailto:bjprager@seas.upenn.edu)

