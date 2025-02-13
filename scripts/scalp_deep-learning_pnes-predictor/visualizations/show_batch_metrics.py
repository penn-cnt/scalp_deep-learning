import pandas as PD
from sys import argv
from pathlib import Path

if __name__ == '__main__':

    # get the list of checkpoints
    files = []
    for path in Path(argv[1]).rglob('*.pth'):
        files.append(str(path))

    # Make a dataframe
    DF             = PD.DataFrame()
    DF['folder']   = [ival.split('/train_pnes_handler_')[1].split('_')[0] for ival in files]
    DF['checknum'] = [ival.split('checkpoint_')[1].split('/')[0] for ival in files]

    print(DF)