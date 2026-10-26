import subprocess,pathlib,time,json,datetime
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
start=time.monotonic()
p=subprocess.run(['/usr/bin/ionice', '-c', '3', '/usr/bin/nice', '-n', '19', '/usr/bin/taskset', '-c', '95', '/usr/bin/time', '-v', '-o', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/closed_D_auxiliary_full_pilot_time_v_20261024T1925.log', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python', '-u', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/qualification/cold_auxiliary_relocation/operate.py', 'pilot', '--plan-sha256', '55ac68a9801609a49f33b68eb533d13cf2308a3efa5b02df191d568b7b4875b5', '--journal', '/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/reports/closed_D_auxiliary_full_pilot_20261024.json'])
(R/'reports/closed_D_auxiliary_full_pilot_process_20261024T1925.json').write_text(json.dumps(dict(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=p.returncode,seconds=time.monotonic()-start),indent=2))
raise SystemExit(p.returncode)
