"""Default main-branch entry point for the conference21 experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'studies/conference21'))
from conference21.__main__ import main
if __name__=='__main__':main()
