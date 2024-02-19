# Data Audit Tool

Due to the increasing data volume, regular data audits are required to remove duplicate data and identify non-essential data to send to deep storage. This tool is meant to help staff interact with audit data in an easier format.

## Layout

![GUI](audit_gui.png)

The above is an example view of the data audit gui. An explanation of each field follows:

- *Search for:* This field accepts alphanumeric input and will search the data audit for any results that (nearly) match the provided string. The behavior of how this string can be used is explained in following fields.
- #Search by:* This radio button sets whether you wish to look for your search entry within the file path, or by md5.
    - Since an md5 value is a unique value for each file, a near match does not mean a nearly similar file.
    - It is recommended to search by a complete md5, or enter an exact substring of a known md5 checksum to find any matches across systems.
