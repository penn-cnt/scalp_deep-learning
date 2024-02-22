# Data Audit Tool

Due to the increasing data volume, regular data audits are required to remove duplicate data and identify non-essential data to send to deep storage. This tool is meant to help staff interact with audit data in an easier format.

## Installation

1. Download and install anaconda.
2. Install the needed python environment using the provided environment file.
    > conda env create --file ./envs/cnt_audit.yml 
3. Activate the conda environment
    > conda activate cnt_audit
4. Run the code
    > python audit.py --search_root `<directory-to_audit>` --outdir `<directory-to-store-audit-files>` --username `<username>` --merge

## Basic Workflow

At a high level, this code is just a wrapper to various shell commands that find data, calculates the checksum, and saves basic information about the file.

To do this effectively, it will recursively search down to each folder and get information on all the **files** in a folder.

This method lets the code *finish* a task in a reasonable timeframe. By doing a single folder at a time, it can save the temporary results to your output directory and update the audit history saying that folder is done.

Once all the audits are done, it will merge the results and remove all the temporary data.

If the code dies halfway through, or you need to restart, it should be able to find where it left off. It will inform you that it is skipping folders it has already completed.

## Sample Output

A sample output audit file can be found [here](interface/modules/samples/sample.audit).

## Commands

For more information about any given input argument, you can always run
> python audit.py --help

A brief summary of each argument follows:
- search_root: Top level directory to recursively search down from for data to audit
    - **NOTE** Due to the sheer volume of data, you might want to avoid pointing to the highest folder available. The code will work, but the initial creation of a file manifest will take a long time. Also, you will be left with a lot of temporary files while it tries to go through everything.
- outdir: Output directory to store temporary files and the final audit information.
    - **NOTE** The code will create this folder if needed. So be careful with typos lest you make an odd new directory path on your system.
- username: A username to be appended to the audit filename. This is so we can track the source of audit files from multiple users at this early stage.
- merge: Once the audit is complete (i.e. has finished going through the file manifest) merge temporary files into a final audit file and remove temp files.
- os: unix/windows backend. Will dictate how the code looks for data. Defaults to unix. 
- cmd_path: A path to the configuration file that tells a system how to find and audit data. This should point to information within 'configs/' in most cases. Defults to the most common architecture for CNT remote systems. (i.e. md5sum hashing)
