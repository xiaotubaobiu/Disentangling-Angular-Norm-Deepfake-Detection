import copy,importlib.util,json,pathlib,pytest
P=pathlib.Path(__file__).parents[1]/'tools/h_parallel_matrix_admission_audit.py'
def module():
 s=importlib.util.spec_from_file_location('generic_H_admission_under_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture(bone='clip_vit_b16',seed=3407):
 m=module();ref=m.read(m.old.REFERENCE);reg=m.read(m.old.ROOT/'H_run_registry.json');accepted=m.read(m.old.ALL3_ACCEPTED);manifest=m.read(m.old.MANIFEST)
 pins=m.reconstruct_pins(bone,ref,manifest,reg,accepted)
 original=m.read(m.old.ROOT/'reports/H_clip_vit_b16_seed42_independent_7role_parallel_candidate_20261025T0645.json')
 c=copy.deepcopy(original);c.update(backbone=bone,seed=seed,pinned_sha256=pins);c['identity']=m.expected_identity(bone,seed,pins,c['identity']['cache_completion_sha256'],ref)
 for row in c['fresh_ledger_invocations']:row.update(backbone=bone,reference_sha256=pins[str(m.old.ROOT/f'reports/H_{bone}_AMP_paired_ledger.json')])
 return m,c,pins,ref

@pytest.mark.parametrize('bone',['clip_vit_b16','dinov2_vit_b14','efficientnet_b0'])
def test_independently_reconstruct_all3_107_inventory(bone):
 m,c,pins,ref=fixture(bone);assert len(pins)==107;assert m.validate_candidate(c,bone,3407,ref,pins)==c['identity']
 assert all(str(m.old.ROOT/'runs'/f'H_formal_{bone}_{h}_seed{s}'/name) in pins for s in m.old.SEEDS for h in ('B','AB') for name in ('identity.json','resolved_config.json','history.json','complete.json','H_completion_seal.json'))
 for path,sha in pins.items(): assert m.old.sha(path)==sha

@pytest.mark.parametrize('change',[lambda c:c.update(seed=True),lambda c:c.update(roles=True),lambda c:c.update(all_source_completions_fresh=True),lambda c:c.update(cross_invocation_receipt_reuse=True),lambda c:c['fresh_ledger_invocations'][0].update(fresh_source_completions=True),lambda c:c['identity'].update(bootstrap_repeats=1999),lambda c:c['identity'].update(source_threshold='target threshold'),lambda c:c['pinned_sha256'].pop(next(iter(c['pinned_sha256']))),lambda c:c['fresh_ledger_invocations'][0].update(reference_sha256='0'*64),lambda c:c['fresh_ledger_invocations'][0].update(elapsed_seconds=float('nan')),lambda c:c['fresh_ledger_invocations'].pop()])
def test_invalid_candidate_rejected(change):
 m,c,pins,ref=fixture();expected=dict(pins);change(c)
 with pytest.raises((AssertionError,ValueError,KeyError)):m.validate_candidate(c,'clip_vit_b16',3407,ref,expected)

def test_reference_clip_inventory_exact_legacy():
 m,c,pins,ref=fixture();assert pins==ref['pinned_sha256']

def test_actual_finite_launches_remain_accepted_metadata():
 m=module()
 for name in ['H_primary_admit_clip_seed3407_slot1_20261025T2000_launch.json','H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030_launch.json']:
  launch=m.read(m.old.ROOT/'reports'/name);m.validate_launch(launch,launch['backbone'],launch['seed'])

def test_live_process_never_terminal():
 import os
 m=module()
 with pytest.raises(AssertionError):m.validate_exit(dict(pid=os.getpid()))

def test_absent_process_can_be_terminal():
 m=module();assert m.validate_exit(dict(pid=999999999))==[999999999]

@pytest.mark.parametrize('change',[lambda x:x.update(CPU_slot=1),lambda x:x.update(parameterized_entry_code=x['parameterized_entry_code']+'print(1)'),lambda x:x.update(seed=42),lambda x:x.update(new_dispatcher=True)])
def test_changed_finite_harness_rejected(change):
 m=module();launch=m.read(m.old.ROOT/'reports/H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030_launch.json');change(launch)
 with pytest.raises((AssertionError,ValueError)):m.validate_launch(launch,launch['backbone'],launch['seed'])

def test_preparation_no_admission():
 m=module();assert m.main([])['production_admission_started'] is False

@pytest.mark.parametrize('change',[lambda x:x['observed_process_tree'][1].update(cmdline='wrong'),lambda x:x['observed_process_tree'][0].update(start_ticks=1),lambda x:x.update(launch_sha256='0'*64),lambda x:x['observed_process_tree'].pop()])
def test_invalid_actual_process_witness_rejected(change):
 m=module();L=m.read(m.old.ROOT/'reports/H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030_launch.json');w=m.read(m.HERE/'H_dinov2_vit_b14_seed7_actual_producer_process_witness.json');change(w)
 with pytest.raises((AssertionError,KeyError)):m.validate_witness(w,L,w.get('launch'),m.sha(w.get('launch')))

@pytest.mark.parametrize('mutation',['early_complete','missing_start','duplicate_complete','exit1','bool_ledger_index'])
def test_bad_terminal_witness_rejected(mutation):
 m=module();b='dinov2_vit_b14';s=7;starts=[f'BEGIN_FRESH_ORIGINAL_LEDGER {n} {b} AMP models10_no_receipt_reuse' for n in range(1,15)];end='PASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT '+str(m.ROOT/f'reports/H_{b}_seed{s}_independent_7role_parallel_candidate_20261025T0645.json');timing='Elapsed (wall clock) time (h:mm:ss or m:ss): 1:23.4\nMaximum resident set size (kbytes): 123456\nExit status: 0\n';lines=starts+[end]
 if mutation=='early_complete':lines=[end]+starts
 elif mutation=='missing_start':lines.pop(0)
 elif mutation=='duplicate_complete':lines.append(end)
 elif mutation=='bool_ledger_index':lines[0]=lines[0].replace(' 1 ',' True ')
 else:timing=timing.replace('status: 0','status: 1')
 with pytest.raises(AssertionError):m.validate_logs('\n'.join(lines),timing,b,s)

@pytest.mark.parametrize('bone',['clip_vit_b16','dinov2_vit_b14','efficientnet_b0'])
def test_real_all3_role_headers_match_sealed_source_metadata(bone):
 m,c,pins,ref=fixture(bone,7);i=c['identity']
 for history,run in i['run_ids'].items():
  r=m.ROOT/'runs'/run;seal=m.read(r/'H_completion_seal.json');ri=m.read(r/'identity.json')
  for role in m.old.ROLES:
   marker=m.read(m.ROOT/'features/H'/run/role/'complete.json');d=m.validate_marker(marker,bone,history,role,i,pins,ref,seal,ri);assert d['rows']>0

@pytest.mark.parametrize('bone,seed',[('clip_vit_b16',3407),('dinov2_vit_b14',7)])
def test_actual_process_witness_exact_executable_chain_positive(monkeypatch,bone,seed):
 m=module();w=m.read(m.HERE/f'H_{bone}_seed{seed}_actual_producer_process_witness.json');L=m.read(w['launch']);monkeypatch.setattr(m,'validate_exit',lambda *args:[x['pid'] for x in w['observed_process_tree']]);assert len(m.validate_witness(w,L,w['launch'],m.sha(w['launch'])))==2

def test_positive_log_timing():
 m=module();bone='dinov2_vit_b14';seed=7;lines=[f'BEGIN_FRESH_ORIGINAL_LEDGER {n} {bone} AMP models10_no_receipt_reuse' for n in range(1,15)];lines.append('PASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT '+str(m.ROOT/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json'));timing='Elapsed (wall clock) time (h:mm:ss or m:ss): 1:23.4\nMaximum resident set size (kbytes): 123456\nExit status: 0\n';assert m.validate_logs('\n'.join(lines),timing,bone,seed)['exit_status']==0

def test_adjacent_tick_contamination_is_not_live_pid_reuse():
 import os,datetime
 m=module();pid=os.getpid();fields=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();w=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_process_tree=[dict(pid=pid,start_ticks=int(fields[19])+1)])
 with pytest.raises(AssertionError):m.validate_exit(dict(pid=pid),w)

def test_DINO_formal_mode_disabled_until_own_real_qualification_chain_exists():
 m=module()
 with pytest.raises(AssertionError):m.validate_mode('dinov2_vit_b14',7,'qualified_pair')
 assert m.validate_mode('dinov2_vit_b14',7,'actual_qualification_prototype') is None
 assert m.validate_mode('clip_vit_b16',3407,'qualified_pair') is None

def test_qualified_pair_explicit_external_Q_proof_required():
 m=module();assert m.expected_refs('qualified_pair')=={'launch','candidate','stdout','time_v','process_witness','CPU_qualification'};assert m.expected_refs('actual_qualification_prototype')=={'launch','candidate','stdout','time_v','process_witness'}
