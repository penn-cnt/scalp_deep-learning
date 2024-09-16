# Codehub

Codehub is a repository designed to help create workflows for data cleaning, preprocessing, and feature extraction. It includes various utility scripts that facilitate acquiring additional data, associating datasets, validating data, and other key tasks involved in building data pipelines.

## Files

### `allowed_arguments.yaml`
This file lists all allowed options and provides help strings for inputs to the workflow. It is essential for validating user inputs and ensuring smooth execution of workflows.

### `code_diagram.py`
A utility script that generates a diagram of all the code in the codehub repository. It also prints documentation for each component, making it easier to visualize and understand the overall structure.

### `epipy.py`
The main script of the repository. It kicks off the workflows and is the front-end for lab workflows.

## Folders

### `components`
This folder contains all the scripts responsible for adding new functionality to the workflows. It also manages the backend code that ensures the workflows operate efficiently.

### `configs`
The `configs` folder holds configuration files used for different workflows. It includes scripts for generating new configuration files, allowing flexible customization of workflows for various use cases.

### `utils`
The `utils` folder contains a variety of scripts that are not part of the core workflow but offer valuable additional functionality for both the workflows and broader data analysis tasks.

## Installation

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus lacinia odio vitae vestibulum. Sed ac felis nec odio convallis aliquet.

## Usage Examples

Placeholder for usage examples.

## Contributing

Contributions are welcome! If you have suggestions or find bugs, feel free to open an issue or submit a pull request. We provide information for how to proceed with two of the most common contribiutions below:

### Epipy 
If adding new functionality to the lab pipeline, code should be added to the **components** folder. More information about the different workflow blocks can be found in the components folder, but each block should contain an internal and public component. Internal components are specific actions epipy takes to facilitate the workflow. These should not be changed lightly.

Public components are where lab code can be saved for everyone to use and for epipy to access. It is formatted to allow for easy importing into interactive shells or python notebooks. New code can be inserted with relative ease as a new python method/function. For more information on how to add code to these libraries, please refer to [our examples folder](https://github.com/penn-cnt/CNT-codehub/tree/main/examples).

### Utility Scripts
The utility scripts are not built into the epipy framework, and do not require specific formatting. To add a utility script, simply identify or create a new folder that generally defines the task being done (data acquisition/data validation/etc.) and add your code to the existing folder for that task type, or create a new folder defining the task and add it there.

### Pull Requests
Submit a pull request to share your changes with the lab as a whole. The data team will review the request before merging it, or sending it back to you for more clarity or bug fixes.