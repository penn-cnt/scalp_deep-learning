# Search Data Cache

This code is meant to acquire the latest data cache report and help the user search for data in the cache. This tool allows users to find out if data already exists in the CNT data cache before downloading data needlessly.

## Installation

You will need the following packages to run the search software. You can run the following commands to install the packages.

```
pip install argparse
pip install paramiko
pip install pandas
pip install prettytable
```

## Example usage

> python data_search.py --username bjprager

The code will by default search in the borel data cache for existing datasets. You will need to provide a username to acquire the new data cache record over ssh protocol, and will be prompted for a password at run time.

For more information, you can run
> python data_search.py --help

## Expected Behavior

This code will create a table of the data available in our data cache. It will display the top 100 results, but can search across the entire cache. You can then query the table to find data that matches your criteria. You will be requested for:

    - subcol : The column name to search within. Defaults to orig_filename. (i.e. iEEG.org filenames)
    - substr : The substring to search for

You will get back a table with all the entries that match your query. You can then use the subject number and session numbers to find the correct dataset within the data cache.

### Sample output

An example output from the script in terminal might look like:
```
python data_search.py --username bjprager
Please enter password for remote system:
Password: 
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
|  orig_filename  |  source  | creator  | gendate  | uid | subject_number | session_number | run_number |    start    |  duration  |
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     1      |  2921875000 |  58398437  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     2      |  3996093750 | 253906250  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     3      |  5738281250 |  46875000  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     5      | 16890625000 | 632812500  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     6      | 24949218750 | 292968750  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     7      | 31011718750 | 433593750  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     8      | 42308593750 | 769531250  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     10     | 60148437500 | 453125000  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     2      | 14839843750 | 683593750  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     5      | 53164062500 | 410156250  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     6      | 60937500000 | 363281250  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     7      | 62976562500 | 175781250  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     8      | 63585937500 | 867187500  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     4      | 13003906250 | 371093750  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     9      | 46804687500 | 800781250  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     11     | 72996093750 | 433589843  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     1      |  1714843750 | 921875000  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     3      | 28746093750 | 1183593750 |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     4      | 44046875000 | 796875000  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     9      | 74402343750 | 382812500  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     10     | 82238281250 | 421871093  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     5      | 19410156250 | 363281250  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     12     | 54921875000 | 390625000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     15     | 85597656250 | 671871093  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     1      |  1199218750 | 117187500  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     2      |  9476562500 | 128906250  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     4      | 20351562500 | 238281250  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     5      | 31867187500 | 1527343750 |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     6      | 55285156250 | 531250000  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     7      | 60726562500 | 312500000  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     8      | 61726562500 | 714843750  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     9      | 67140625000 | 695312500  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     10     | 69777343750 | 562500000  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     11     | 76214843750 | 562496093  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     1      |  105468750  | 238281250  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     2      |  4445312500 | 339843750  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     3      |  6613281250 | 257812500  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     4      | 12519531250 | 187500000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     7      | 26812500000 |  31250000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     8      | 35722656250 | 1648437500 |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     9      | 38906250000 | 964843750  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     10     | 44554687500 | 402343750  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     13     | 63000000000 | 730468750  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     14     | 73812500000 | 2812500000 |
| EMU1169_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     3      | 10875000000 | 203125000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     6      | 26375000000 | 187500000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     11     | 51875000000 | 953125000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     1      |  601562500  | 269531250  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     3      | 10707031250 | 394531250  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     4      | 14816406250 | 562500000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     5      | 19242187500 | 453125000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     7      | 29585937500 | 578125000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     9      | 52890625000 | 1488281250 |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     10     | 56093750000 | 1007812500 |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     11     | 57679687500 | 796875000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     12     | 60757812500 | 1992187500 |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     1      |  585937500  | 121093750  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     3      | 19613281250 | 449218750  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     4      | 45398437500 | 398437500  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     6      | 71687500000 | 1078125000 |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     7      | 81769531250 | 605464843  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     2      |  5898437500 | 339843750  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     6      | 22261718750 | 503906250  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     8      | 43796875000 | 1140625000 |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     14     | 73878906250 | 1062500000 |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     15     | 78839843750 | 851558593  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     2      |  4949218750 | 351562500  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     5      | 65511718750 | 656250000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     13     | 71175781250 | 1671875000 |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     1      |  3976562500 | 242187500  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     2      | 19238281250 | 589843750  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     3      | 37496093750 | 132812500  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     4      | 50289062500 | 578125000  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     6      | 72199218750 | 785156250  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     8      | 85308593750 | 812496093  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     1      |  160156250  | 421875000  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     2      |  8863281250 | 351562500  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     3      | 29593750000 | 242187500  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     4      | 36738281250 | 421875000  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     5      | 49375000000 | 468750000  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     9      | 81800781250 | 531246093  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     5      | 60355468750 | 453125000  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     6      | 59957031250 | 269531250  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     7      | 68652343750 | 492187500  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     8      | 76523437500 | 644531250  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     7      | 78535156250 | 769531250  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     1      |  3312500000 | 921875000  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     5      | 25609375000 | 316406250  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     9      | 70187500000 | 953125000  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     10     | 74027343750 | 601562500  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     11     | 80894531250 | 621089843  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     1      |  9445312500 | 148437500  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     2      | 15761718750 | 1035156250 |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     3      | 19875000000 | 812500000  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     4      | 31750000000 | 753906250  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     7      | 59703125000 | 242187500  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     8      | 66355468750 | 628906250  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     9      | 82421875000 | 671871093  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       5        |     2      |  5535156250 | 570312500  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       5        |     3      |  9867187500 | 585937500  |
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
Search for entry (y/n) or reset table (R/r)? y
Please enter column to search within (default='orig_filename'): 
Please enter substring to search for: 2014
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
|  orig_filename  |  source  | creator  | gendate  | uid | subject_number | session_number | run_number |    start    |  duration  |
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     1      |  7093750000 | 457031250  |
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     2      | 11492187500 | 500000000  |
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     3      | 13363281250 | 1136718750 |
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     4      | 17199218750 | 1757812500 |
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     5      | 28664062500 | 648242187  |
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     6      | 29369863281 | 1063730468 |
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     7      | 38734375000 | 3476562500 |
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     8      | 53171875000 | 2328125000 |
| EMU2014_Day01_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       1        |     9      | 61523437500 | 730464843  |
| EMU2014_Day02_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       2        |     2      |  4011718750 | 699218750  |
| EMU2014_Day02_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       2        |     3      | 10371093750 | 925781250  |
| EMU2014_Day02_1 | ieeg.org | bjprager | 07-05-24 | 3.0 |      3.0       |       2        |     4      | 23000000000 | 1499996093 |
| EMU2014_Day02_1 | ieeg.org | bjprager | 08-05-24 | 3.0 |      3.0       |       2        |     1      |  421875000  | 925781250  |
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
```

You can reset the table by pressing `r`. If we reset and now want to look by subject number and a particular session, for example, we can do the following:
```
Search for entry (y/n) or reset table (R/r)? r
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
|  orig_filename  |  source  | creator  | gendate  | uid | subject_number | session_number | run_number |    start    |  duration  |
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     1      |  2921875000 |  58398437  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     2      |  3996093750 | 253906250  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     3      |  5738281250 |  46875000  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     5      | 16890625000 | 632812500  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     6      | 24949218750 | 292968750  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     7      | 31011718750 | 433593750  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     8      | 42308593750 | 769531250  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     10     | 60148437500 | 453125000  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     2      | 14839843750 | 683593750  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     5      | 53164062500 | 410156250  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     6      | 60937500000 | 363281250  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     7      | 62976562500 | 175781250  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       1        |     8      | 63585937500 | 867187500  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     4      | 13003906250 | 371093750  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     9      | 46804687500 | 800781250  |
| EMU0562_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     11     | 72996093750 | 433589843  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     1      |  1714843750 | 921875000  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     3      | 28746093750 | 1183593750 |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     4      | 44046875000 | 796875000  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     9      | 74402343750 | 382812500  |
| EMU1169_Day01_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       1        |     10     | 82238281250 | 421871093  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     5      | 19410156250 | 363281250  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     12     | 54921875000 | 390625000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     15     | 85597656250 | 671871093  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     1      |  1199218750 | 117187500  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     2      |  9476562500 | 128906250  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     4      | 20351562500 | 238281250  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     5      | 31867187500 | 1527343750 |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     6      | 55285156250 | 531250000  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     7      | 60726562500 | 312500000  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     8      | 61726562500 | 714843750  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     9      | 67140625000 | 695312500  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     10     | 69777343750 | 562500000  |
| EMU1169_Day02_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       2        |     11     | 76214843750 | 562496093  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     1      |  105468750  | 238281250  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     2      |  4445312500 | 339843750  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     3      |  6613281250 | 257812500  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     4      | 12519531250 | 187500000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     7      | 26812500000 |  31250000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     8      | 35722656250 | 1648437500 |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     9      | 38906250000 | 964843750  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     10     | 44554687500 | 402343750  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     13     | 63000000000 | 730468750  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     14     | 73812500000 | 2812500000 |
| EMU1169_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     3      | 10875000000 | 203125000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     6      | 26375000000 | 187500000  |
| EMU0562_Day02_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       2        |     11     | 51875000000 | 953125000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     1      |  601562500  | 269531250  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     3      | 10707031250 | 394531250  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     4      | 14816406250 | 562500000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     5      | 19242187500 | 453125000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     7      | 29585937500 | 578125000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     9      | 52890625000 | 1488281250 |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     10     | 56093750000 | 1007812500 |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     11     | 57679687500 | 796875000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     12     | 60757812500 | 1992187500 |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     1      |  585937500  | 121093750  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     3      | 19613281250 | 449218750  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     4      | 45398437500 | 398437500  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     6      | 71687500000 | 1078125000 |
| EMU1169_Day03_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       3        |     7      | 81769531250 | 605464843  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     2      |  5898437500 | 339843750  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     6      | 22261718750 | 503906250  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     8      | 43796875000 | 1140625000 |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     14     | 73878906250 | 1062500000 |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     15     | 78839843750 | 851558593  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     2      |  4949218750 | 351562500  |
| EMU1169_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     5      | 65511718750 | 656250000  |
| EMU0562_Day03_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       3        |     13     | 71175781250 | 1671875000 |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     1      |  3976562500 | 242187500  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     2      | 19238281250 | 589843750  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     3      | 37496093750 | 132812500  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     4      | 50289062500 | 578125000  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     6      | 72199218750 | 785156250  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     8      | 85308593750 | 812496093  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     1      |  160156250  | 421875000  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     2      |  8863281250 | 351562500  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     3      | 29593750000 | 242187500  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     4      | 36738281250 | 421875000  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     5      | 49375000000 | 468750000  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       4        |     9      | 81800781250 | 531246093  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     5      | 60355468750 | 453125000  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     6      | 59957031250 | 269531250  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     7      | 68652343750 | 492187500  |
| EMU1169_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     8      | 76523437500 | 644531250  |
| EMU0562_Day04_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       4        |     7      | 78535156250 | 769531250  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     1      |  3312500000 | 921875000  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     5      | 25609375000 | 316406250  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     9      | 70187500000 | 953125000  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     10     | 74027343750 | 601562500  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     11     | 80894531250 | 621089843  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     1      |  9445312500 | 148437500  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     2      | 15761718750 | 1035156250 |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     3      | 19875000000 | 812500000  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     4      | 31750000000 | 753906250  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     7      | 59703125000 | 242187500  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     8      | 66355468750 | 628906250  |
| EMU1169_Day05_1 | ieeg.org | bjprager | 07-05-24 | 0.0 |      0.0       |       5        |     9      | 82421875000 | 671871093  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       5        |     2      |  5535156250 | 570312500  |
| EMU0562_Day05_1 | ieeg.org | bjprager | 08-05-24 | 0.0 |      0.0       |       5        |     3      |  9867187500 | 585937500  |
+-----------------+----------+----------+----------+-----+----------------+----------------+------------+-------------+------------+
Search for entry (y/n) or reset table (R/r)? y
Please enter column to search within (default='orig_filename'): subject_number
Please enter substring to search for: 108
+-----------------+----------+----------+----------+-------+----------------+----------------+------------+-------------+------------+
|  orig_filename  |  source  | creator  | gendate  |  uid  | subject_number | session_number | run_number |    start    |  duration  |
+-----------------+----------+----------+----------+-------+----------------+----------------+------------+-------------+------------+
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     1      |  2000000000 | 339843750  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     2      |  3453125000 | 562500000  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     4      | 11121093750 | 535156250  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     5      | 17949218750 | 535156250  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     8      | 27792968750 | 460937500  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     9      | 36007812500 | 414062500  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     10     | 45660156250 | 851562500  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     11     | 60828125000 | 824218750  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     12     | 67816406250 | 308589843  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     6      | 18750000000 | 570312500  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     7      | 21027343750 | 191406250  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     1      |  2843750000 | 253906250  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     2      |  3777343750 | 652343750  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     3      | 11976562500 | 343750000  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     4      | 14324218750 | 207031250  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     5      | 17871093750 | 285156250  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     6      | 28449218750 | 589843750  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     7      | 30300781250 | 773437500  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     8      | 44472656250 | 562500000  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     11     | 59132812500 | 468750000  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     12     | 61980468750 | 488281250  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     9      | 52164062500 | 160156250  |
| EMU1519_Day01_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       1        |     3      |  5546875000 | 382812500  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       1        |     13     | 63878906250 | 421875000  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       1        |     14     | 65632812500 | 204128906  |
| EMU1520_Day01_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       1        |     10     | 58437500000 | 453125000  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     1      |  136718750  | 757812500  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     2      |  3898437500 | 507812500  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     3      |  7453125000 | 800781250  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     5      | 20550781250 | 360511718  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     7      | 33550781250 | 628906250  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     11     | 84914062500 | 593746093  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     6      | 20913609375 | 648890625  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     8      | 47347656250 | 808593750  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     2      |   7812500   | 433593750  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     4      |  1714843750 | 296875000  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     6      |  7050781250 | 394531250  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     10     | 29281250000 |  58593750  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     16     | 32042968750 |  35156250  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     18     | 33195312500 | 406250000  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     28     | 49449218750 |  58593750  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     30     | 53429687500 | 117187500  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     32     | 64441406250 | 683593750  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     38     | 77207031250 | 175781250  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     40     | 77746093750 | 574214843  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     8      | 26218750000 | 531250000  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     12     | 29808593750 | 511718750  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     14     | 31289062500 |  46875000  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     26     | 46164062500 | 597656250  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     34     | 66679687500 | 296875000  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       2        |     36     | 75976562500 |  66406250  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       2        |     4      | 14394531250 | 742187500  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       2        |     9      | 63816406250 | 203125000  |
| EMU1519_Day02_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       2        |     10     | 70867187500 | 1324218750 |
| EMU1520_Day02_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       2        |     1      |     3906    |  7808594   |
| EMU1520_Day02_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       2        |     20     | 34230468750 |  46875000  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       2        |     22     | 39011718750 | 453125000  |
| EMU1520_Day02_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       2        |     24     | 41304687500 | 238281250  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     1      |  156250000  | 589843750  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     3      |  1542968750 | 355468750  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     4      |  1945312500 | 425781250  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     5      |  3839843750 | 218750000  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     6      | 10320312500 | 128906250  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     7      | 18746093750 | 339843750  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     8      | 34300781250 |  74218750  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     9      | 40312500000 | 296875000  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     10     | 45578125000 | 726562500  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     11     | 59320312500 | 246093750  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     12     | 67761718750 | 699218750  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     13     | 73578125000 | 136718750  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     14     | 73812500000 | 1148437500 |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     15     | 75191406250 | 457031250  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     16     | 82550781250 | 367187500  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     17     | 85644531250 | 687496093  |
| EMU1519_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     2      |  1062500000 |  3906250   |
| EMU1520_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     4      | 16421875000 | 386718750  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     6      | 27578125000 | 484375000  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     12     | 34011718750 | 281250000  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     14     | 45832031250 | 375000000  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     16     | 54042968750 | 406250000  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     22     | 77617187500 | 531246093  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     2      |  8222656250 | 589843750  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       3        |     8      | 30988281250 | 207031250  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       3        |     10     | 31488281250 | 285156250  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       3        |     20     | 68164062500 | 480468750  |
| EMU1520_Day03_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       3        |     18     | 60378906250 | 421875000  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     1      |   15625000  | 835937500  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     2      |  1234375000 | 613281250  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     3      |  2515625000 | 199218750  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     4      |  3703125000 | 210937500  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     5      | 12816406250 | 1167968750 |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     7      | 48148437500 | 1367187500 |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     10     | 67574218750 | 441406250  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     11     | 68085937500 | 175781250  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     13     | 84265625000 | 656246093  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     8      | 63121093750 | 1519531250 |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     9      | 66472656250 | 960937500  |
| EMU1519_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     12     | 78777343750 | 132812500  |
| EMU1520_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     2      | 13628906250 | 550781250  |
| EMU1520_Day04_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       4        |     6      | 27750000000 | 589843750  |
+-----------------+----------+----------+----------+-------+----------------+----------------+------------+-------------+------------+
Search for entry (y/n) or reset table (R/r)? y
Please enter column to search within (default='orig_filename'): session_number
Please enter substring to search for: 1
+-----------------+----------+----------+----------+-------+----------------+----------------+------------+-------------+-----------+
|  orig_filename  |  source  | creator  | gendate  |  uid  | subject_number | session_number | run_number |    start    |  duration |
+-----------------+----------+----------+----------+-------+----------------+----------------+------------+-------------+-----------+
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     1      |  2000000000 | 339843750 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     2      |  3453125000 | 562500000 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     4      | 11121093750 | 535156250 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     5      | 17949218750 | 535156250 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     8      | 27792968750 | 460937500 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     9      | 36007812500 | 414062500 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     10     | 45660156250 | 851562500 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     11     | 60828125000 | 824218750 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     12     | 67816406250 | 308589843 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     6      | 18750000000 | 570312500 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     7      | 21027343750 | 191406250 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     1      |  2843750000 | 253906250 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     2      |  3777343750 | 652343750 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     3      | 11976562500 | 343750000 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     4      | 14324218750 | 207031250 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     5      | 17871093750 | 285156250 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     6      | 28449218750 | 589843750 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     7      | 30300781250 | 773437500 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     8      | 44472656250 | 562500000 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     11     | 59132812500 | 468750000 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     12     | 61980468750 | 488281250 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 07-05-24 | 108.0 |     108.0      |       1        |     9      | 52164062500 | 160156250 |
| EMU1519_Day01_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       1        |     3      |  5546875000 | 382812500 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       1        |     13     | 63878906250 | 421875000 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       1        |     14     | 65632812500 | 204128906 |
| EMU1520_Day01_1 | ieeg.org | bjprager | 08-05-24 | 108.0 |     108.0      |       1        |     10     | 58437500000 | 453125000 |
+-----------------+----------+----------+----------+-------+----------------+----------------+------------+-------------+-----------+
```