# Basic Python Imports
import sys
import argparse
from glob import glob

# Local imports
from modules.channel_mapping import *
from modules.channel_clean import *
from modules.channel_montage import *

class CustomFormatter(argparse.HelpFormatter):
    """
    Custom formatting class to get a better argument parser help output.
    """

    def _split_lines(self, text, width):
        if text.startswith("R|"):
            return text[2:].splitlines()
        return super()._split_lines(text, width)

def make_help_str(idict):
    """
    Make a well-formated help string for the possible keyword mappings

    Args:
        idict (dict): Dictionary containing the allowed keywords values and their explanation.

    Returns:
        str: Formatted help string
    """

    return "\n".join([f"{key:15}: {value}" for key, value in idict.items()])

if __name__ == '__main__':

    # Argument option creation
    allowed_channel_args    = {'HUP1020': "Channels associated with a 10-20 montage performed at HUP.",
                               'RAW': "Use all possible channels. Warning, channels may not match across different datasets."}
    allowed_montage_args    = {'HUP1020': "Use a 10-20 montage.",
                               'COMMON_AVERAGE': "Use a common average montage."}
    
    # Pretty argument string creation
    allowed_channel_help   = make_help_str(allowed_channel_args)
    allowed_montage_help   = make_help_str(allowed_montage_args)

    # Command line options needed to obtain data.
    parser = argparse.ArgumentParser(description="Simplified data merging tool.", formatter_class=CustomFormatter)

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--file", type=str, help="Input file to plot.")
    input_group.add_argument("--wildcard", type=str, help="Wildcard enabled path to plot multiple datasets.")

    prep_group = parser.add_argument_group('Data preparation options')
    prep_group.add_argument("--channel_list", choices=list(allowed_channel_args.keys()), default="HUP1020", help=f"R|Choose an option:\n{allowed_channel_help}")

    args = parser.parse_args()

    # Create the file list to read in
    if args.file != None:
        files = [args.file]
    else:
        files = glob(args.wildcard)

