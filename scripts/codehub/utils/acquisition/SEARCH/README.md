# Search Data Cache

This code is meant to acquire the latest data cache report and help the user search for data in the cache. This tool allows users to find out if data already exists in the CNT data cache before downloading data needlessly.

## Example usage

> python data_search.py --username bjprager

The code will by default search in the borel data cache for existing datasets. You will need to provide a username to acquire the new data cache record over ssh protocol, and will be prompted for a password at run time.

For more information, you can run
> python data_search.py --help
