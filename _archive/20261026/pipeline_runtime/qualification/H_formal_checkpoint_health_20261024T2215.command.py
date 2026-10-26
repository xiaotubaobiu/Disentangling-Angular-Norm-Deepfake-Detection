# Reuse verified auditor unchanged; only the report destination differs.
import importlib.util,hashlib
from pathlib import Path
s=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/qualification/H_formal_checkpoint_health_20261024T2145.py')
assert hashlib.sha256(s.read_bytes()).hexdigest()=='c59bd1514efa1892027bab7910d25ff6a3057efaae82d3aff988e85a12058cc6'
spec=importlib.util.spec_from_file_location('h_atomic_snapshot_2215',s)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.REPORT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/H_formal_checkpoint_health_20261024T2215.json')
module.main()
