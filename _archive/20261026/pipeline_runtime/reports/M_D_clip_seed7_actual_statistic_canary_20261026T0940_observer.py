import pathlib,json,os,sys,subprocess,resource,hashlib,datetime,time,re
c=json.loads(pathlib.Path(sys.argv[1]).read_text());base=pathlib.Path(sys.argv[2]);env=dict(os.environ,**c['environment']);cap=c['address_space_limit_bytes']
def limited():resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(p,d):
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with open(c['stdout'],'xb') as log:
 p=subprocess.Popen(c['command'],cwd=c['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limited)
 limits=[];child=None
 for _ in range(50):
  path=pathlib.Path(f'/proc/{p.pid}/task/{p.pid}/children')
  if path.exists() and path.read_text().split():child=int(path.read_text().split()[0]);break
  if p.poll() is not None:break
  time.sleep(.02)
 for pid in (p.pid,child):
  if pid is not None:
   path=pathlib.Path(f'/proc/{pid}/limits')
   if path.exists():limits.append({'pid':pid,'Max_address_space':next(x for x in path.read_text().splitlines() if x.startswith('Max address space'))})
 launch=dict(c,observer_pid=os.getpid(),wrapper_pid=p.pid,initial_python_child_pid=child,actual_proc_limit_readback=limits);lp=pathlib.Path(str(base)+'_launch.json');save(lp,launch);rc=p.wait()
 timing=pathlib.Path(c['time_v']).read_text();zero=(len(re.findall(r'^\s*Exit status: 0\s*$',timing,re.M))==1)
 result={'schema':'M_STATISTIC_PRODUCER_EXIT_V1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':c['job'],'launch_sha256':sha(lp),'exit_code':rc,'returncode_observed_by_wait':True,'wrapper_and_child_true_exit_observed':bool(rc==0 and zero),'launch':{'path':str(lp),'sha256':sha(lp)},'stdout':{'path':c['stdout'],'sha256':sha(c['stdout'])},'time_v':{'path':c['time_v'],'sha256':sha(c['time_v'])},'actual_resource_and_independent_numeric_acceptance':False}
 save(pathlib.Path(str(base)+'_exit.json'),result)
