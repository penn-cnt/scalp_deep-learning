from sys import argv
from pathlib import Path

if __name__ == '__main__':

    # get the list of checkpoints
    files = []
    for path in Path(argv[1]).rglob('*check*'):
        files.append(path)
    print(files)