# Data Audit Tool

Due to the increasing data volume, regular data audits are required to remove duplicate data and identify non-essential data to send to deep storage. This tool is meant to help staff interact with audit data in an easier format.

## Audit Fields

The audit tracks five individual fields for the end-user to explore:

1. **path** The fullpath to a file on the given computer system.

## Layout

![GUI](audit_gui.png)

The above is an example view of the data audit gui. An explanation of each field follows:

- **Search for:** This field accepts alphanumeric input and will search the data audit for any results that (nearly) match the provided string. The behavior of how this string can be used is explained in following fields.
- **Search by:** This radio button sets whether you wish to look for your search entry within the file path, or by md5.
    - Since an md5 value is a unique value for each file, a near match does not mean a nearly similar file.
    - It is recommended to search by a complete md5, or enter an exact substring of a known md5 checksum to find any matches across systems.
- **Fuzzy Match:** This radio button toggles fuzzy string matching filenames. Fuzzy matching is a means of finding nearly similar strings, and can be used if you expect the filenames you are searching for may be close to your inputted string, but not exact.
    - This option is slower than an exact match. Use only as needed.
- **Apply to:** Apply the above settings to the data tables for the selected computer system.
- **Shrink Path:** Due to the long filepaths, this shrinks/collapses the folders in the filepath. The larger the number, the more top level folders are hidden from the display.
- **Sort by:** Sort the table values by the selected field. Clicking multiple times toggles between ascending and descending values.
