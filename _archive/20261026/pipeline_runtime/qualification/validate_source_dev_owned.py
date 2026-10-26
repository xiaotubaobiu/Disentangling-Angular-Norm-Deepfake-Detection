from pathlib import Path
import json,sys,runpy
from reliability_v2.trainer import owned_gpu
root=Path(sys.argv[1]); backbone=sys.argv[2]
config=json.loads((root/"configs/base.json").read_text())
with owned_gpu(root,config):
 sys.argv=[str(root/"qualification/validate_source_dev.py"),str(root),backbone]
 runpy.run_path(sys.argv[0],run_name="__main__")
