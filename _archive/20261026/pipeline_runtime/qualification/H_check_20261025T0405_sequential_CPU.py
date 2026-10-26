import subprocess,json,datetime
from pathlib import Path
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
rp=R/"reports"
commands=[['/usr/bin/time', '-v', '-o', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/H_DINO_seed3407_terminal_time_v_20261025T0405.log', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python', '-B', '-u', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/qualification/audit_H_formal_terminal.py', '--run-id', 'H_formal_dinov2_vit_b14_B_seed3407', '--run-id', 'H_formal_dinov2_vit_b14_AB_seed3407', '--report', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/H_DINO_seed3407_terminal_acceptance_20261025T0405.json'], ['/usr/bin/time', '-v', '-o', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/H_formal_checkpoint_health_time_v_20261025T0405.log', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python', '-B', '-u', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/qualification/H_formal_checkpoint_health_20261025T0405.command.py']]
logs=['/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/H_DINO_seed3407_terminal_20261025T0405.log', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/H_formal_checkpoint_health_20261025T0405.log']
for cmd,log in zip(commands,logs):
 with Path(log).open("x") as stream:
  p=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
 if p.returncode:raise SystemExit(p.returncode)
(rp/"H_check_20261025T0405_CPU_sequence_complete.json").write_text(json.dumps({"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"status":"BOTH_SEQUENTIAL_CPU_AUDITS_EXIT0_NO_TRAINING_MUTATION"},indent=2)+"\n")
