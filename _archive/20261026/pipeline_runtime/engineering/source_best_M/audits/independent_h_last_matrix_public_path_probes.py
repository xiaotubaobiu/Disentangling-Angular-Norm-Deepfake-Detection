import copy,json,hashlib,sys
from pathlib import Path
import independent_h_last_matrix_public_fixture as r
f=r.Fixture('path_probes');m=f.m;original=json.loads(f.bp.read_text());results=[]
def probe(name,change,expected_reject):
 b=copy.deepcopy(original);change(b);r.dump(f.bp,b)
 try:m.load_spec(f.bp,r.H(f.bp));out='ACCEPT'
 except (AssertionError,ValueError) as e:out='REJECT'
 results.append(dict(name=name,result=out,expected='REJECT' if expected_reject else 'ACCEPT'))
 assert (out=='REJECT')==expected_reject,(name,out)
probe('public_valid_spec',lambda b:None,False)
try:m.load_spec(f.bp,'0'*64);raise RuntimeError('bad external SHA accepted')
except AssertionError:results.append(dict(name='external_binding_SHA_drift',result='REJECT',expected='REJECT'))
probe('foreign_root_field',lambda b:b.update(root='/tmp/unbound'),True)
probe('outside_input_namespace',lambda b:b['proofs']['cache_audit'].update(path='/tmp/outside_namespace.json'),True)
probe('cross_seed_identity_path',lambda b:b['proofs']['identity'].update(path=str(f.root/'analysis/H/clip_vit_b16/seed2026/identity.json')),True)
source=f.refs['source_manifest'];backup=source.with_suffix('.backup');source.rename(backup);source.symlink_to(backup)
probe('symlink_proof',lambda b:None,True);source.unlink();backup.rename(source)
r.dump(f.bp,original);f.execute()
try:m.main(['--execute','--binding-spec',str(f.bp),'--binding-spec-sha',r.H(f.bp),'--terminal-sha',f.terminal_sha,'--slot','1']);raise RuntimeError('existing output overwritten')
except AssertionError:results.append(dict(name='preserve_prior_output',result='REJECT',expected='REJECT'))
assert len(f.calls)==8
print(json.dumps(results,indent=2));(r.OUT/'public_path_results.json').write_text(json.dumps(results,indent=2)+'\n')
