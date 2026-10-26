#!/usr/bin/env python3
"""One project-root anchored entry; run from any directory."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parent
PY=ROOT/'.venv/bin/python'
sys.path.insert(0,str(ROOT/'src'))
from deepfake_detector.launcher import needs_reexec
if needs_reexec(sys.prefix,ROOT/'.venv'):
    os.execv(str(PY),[str(PY),'-B',str(Path(__file__).resolve()),*sys.argv[1:]])
from deepfake_detector.cli import main
main(ROOT)
