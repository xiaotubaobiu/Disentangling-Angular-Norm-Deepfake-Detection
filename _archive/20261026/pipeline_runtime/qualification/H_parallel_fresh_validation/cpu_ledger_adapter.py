"""Fresh two-thread original ledgers for NEW CPU processes only.

No original source edit, cache owner integration, global receipt memoization,
model/threshold change or scientific validation omission is permitted here.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib,json,threading,time
from parallel_source import collect_fresh,FreshLookup,assert_same_ledger
_INSTALL_LOCK=threading.RLock()
_INSTALLED=set()

def _digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@contextmanager
def installed_fresh_audit(queue_module,config_path,fresh_verify,run_ids_for,reference_for,pinned_sha256):
 config_path=Path(config_path).resolve()
 pins={str(Path(p).resolve()):v for p,v in pinned_sha256.items()}
 if str(config_path) not in pins or not pins:raise ValueError('configuration SHA binding required')
 def check():
  if any(_digest(p)!=v for p,v in pins.items()):raise ValueError('CPU adapter pinned source/input changed')
 check();key=id(queue_module)
 with _INSTALL_LOCK:
  if key in _INSTALLED:raise ValueError('CPU ledger adapter already installed')
  original=queue_module.audit_paired_ledgers;_INSTALLED.add(key)
 receipts=[];call_lock=threading.RLock()
 def adapted(path,backbone,*,precision='AMP',verify=None):
  with call_lock:
   if Path(path).resolve()!=config_path or verify is not None:raise ValueError('CPU adapter context or caller verifier mismatch')
   if queue_module.audit_paired_ledgers is not adapted:raise ValueError('CPU adapter original binding changed')
   check();reference_path=Path(reference_for(backbone,precision)).resolve()
   if str(reference_path) not in pins:raise ValueError('reference ledger must be SHA pinned')
   reference=json.loads(reference_path.read_text())
   ids=list(run_ids_for(backbone,precision))
   if len(ids)!=10 or len(set(ids))!=10:raise ValueError('all exact five pairs required')
   start=time.monotonic();fresh=collect_fresh(config_path,ids,fresh_verify,workers=2)
   check();lookup=FreshLookup(config_path,fresh)
   result=original(config_path,backbone,precision=precision,verify=lookup);lookup.finish()
   assert_same_ledger(reference,result);check()
   receipts.append(dict(backbone=backbone,precision=precision,fresh_source_completions=10,
       exact_original_ledger_except_top_utc=True,reference_sha256=_digest(reference_path),
       elapsed_seconds=time.monotonic()-start,cross_invocation_receipt_reuse=False))
   return result
 try:
  queue_module.audit_paired_ledgers=adapted
  yield receipts
 finally:
  with call_lock:
   changed=queue_module.audit_paired_ledgers is not adapted
   queue_module.audit_paired_ledgers=original
   with _INSTALL_LOCK:_INSTALLED.discard(key)
   if changed:raise ValueError('CPU adapter binding changed during installed scope')
 check()
