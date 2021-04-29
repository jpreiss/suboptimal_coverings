from pathlib import Path
import sys

if __name__ == "__main__":
    outpath = sys.argv[1]
    Path(outpath).touch()
