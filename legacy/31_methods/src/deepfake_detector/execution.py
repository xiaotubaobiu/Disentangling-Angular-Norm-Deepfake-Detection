"""One owned supervisor for qualification, pilot, formal queue and final reporting."""
from pathlib import Path
import json,os,datetime,traceback,time
from .manifest import save_json,digest
from .orchestration import exclusive_lock,termination_guard,container_state,code_hash

def execute(config,project,config_path):
    root=Path(config['artifact_root']);status=root/'execution_status.json'
    with exclusive_lock(root/'execution.lock'),termination_guard():
        record={'status':'RUNNING','owner_pid':os.getpid(),'started':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'code_sha256':code_hash(project),'freeze_sha256':digest(root/'data/freeze_complete.json'),'stages':{}}
        def stage(name,call):
            record['phase']=name;save_json(status,record)
            print(json.dumps({'phase':name,'state':'START','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
            result=call();record['stages'][name]={'status':'PASS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            save_json(status,record);print(json.dumps({'phase':name,'state':'PASS'}),flush=True);return result
        try:
            from .orchestration import preflight
            from .qualification import probe
            from .pilot import run_pilot
            from .queueing import run_queue
            from .reporting import report
            stage('CPU_preflight',lambda:preflight(config,project))
            stage('GPU_qualification',lambda:probe(config,project,config_path))
            stage('end_to_end_pilot',lambda:run_pilot(config,project))
            stage('15_formal_runs_and_features',lambda:run_queue(config,project,config_path))
            stage('30_matched_heads_A_B_C_D_and_report',lambda:report(config,project))
            journal=json.loads((root/'service_state.json').read_text())
            if journal['original']['state']['Running']:
                current=container_state()
                if current['id']!=journal['original']['id'] or not current['state']['Running']:raise ValueError('Original LLM container restoration not confirmed')
                import urllib.request
                with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=15) as response:
                    if response.status!=200:raise ValueError('Original LLM health endpoint not ready')
                record['service_restoration']='original container running; local health HTTP200'
            else:record['service_restoration']='originally stopped; left stopped'
            record.update(status='COMPLETE',phase='COMPLETE',finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save_json(status,record);return record
        except BaseException as error:
            record.update(status='FAILED',failure=type(error).__name__+': '+str(error),finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
            save_json(status,record);raise
