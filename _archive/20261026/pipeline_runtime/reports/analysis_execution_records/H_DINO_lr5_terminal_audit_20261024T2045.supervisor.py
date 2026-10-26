import subprocess,time,json,datetime,pathlib
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
start=time.monotonic()
p=subprocess.run(['/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/environment/bin/python','-u','/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/qualification/H_DINO_lr5_terminal_audit_20261024T2010.py'])
(R/'reports/H_DINO_lr5_terminal_audit_process_20261024T2045.json').write_text(json.dumps(dict(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),returncode=p.returncode,seconds=time.monotonic()-start),indent=2))
raise SystemExit(p.returncode)
