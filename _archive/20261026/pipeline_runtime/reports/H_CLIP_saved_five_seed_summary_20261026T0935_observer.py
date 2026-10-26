import pathlib,json,os,sys,subprocess,resource,hashlib,datetime
c=json.loads(pathlib.Path(sys.argv[1]).read_text());base=pathlib.Path(sys.argv[2]);env=dict(os.environ,**c['environment']);cap=c['address_space_limit_bytes']
def limited():resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
with open(c['stdout'],'xb') as log:
 p=subprocess.Popen(c['command'],cwd=c['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limited)
 launch=dict(c,observer_pid=os.getpid(),wrapper_pid=p.pid);lp=pathlib.Path(str(base)+'_launch.json');lp.write_text(json.dumps(launch,ensure_ascii=False,indent=2)+'\n');rc=p.wait()
result={'schema':'H_SAVED_FIVE_SEED_SUMMARY_TRUE_EXIT_V1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':rc,'returncode_observed_by_wait':True,'launch':{'path':str(lp),'sha256':sha(lp)},'stdout':{'path':c['stdout'],'sha256':sha(c['stdout'])},'time_v':{'path':c['time_v'],'sha256':sha(c['time_v'])},'independent_table_acceptance':False}
pathlib.Path(str(base)+'_exit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
