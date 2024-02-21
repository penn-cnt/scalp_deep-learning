import argparse
import numpy as np

if __name__ == '__main__':

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="iEEG to bids conversion tool.")
    parser.add_argument("--yaml_file", type=str, required=True, help="Yaml file to duplicate into grid format.")
    parser.add_argument("--grid_file", type=str, required=True, help="Yaml grid file with rules for duplication.")
    parser.add_argument("--outdir", type=str, required=True, help="Path to output directory to make grid structure in.")
    parser.add_argument("--cmd", type=str, help="File containing the command line call to the pipeline. Will create updated commands.")
    args = parser.parse_args()