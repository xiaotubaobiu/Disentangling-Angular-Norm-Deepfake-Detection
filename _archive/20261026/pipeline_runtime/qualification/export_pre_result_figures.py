"""CPU preparation figures from accepted v2 metadata and source DEV only."""
from pathlib import Path
import os,json,hashlib,datetime,shutil,fcntl,stat
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
OUT=R/'manuscript_figures/pre_result'
COVERAGE=R/'reports/H_frozen_coverage_repaired_20261024T0320.json'
ACCEPT=R/'reports/H_coverage_T1_table_independent_acceptance_20261024T0320.json'
AUDIT=R/'reports/D_completed_artifact_audit_20261024T1230.json'
CONFIG=R/'configs/execution.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def safe(p):
 p=Path(p);assert p.resolve().is_relative_to(R)
 for a in [p,*p.parents]:
  assert not a.is_symlink(),str(a)
  if a==R:break
 return p

def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 for p in [COVERAGE,ACCEPT,AUDIT,CONFIG]:safe(p);assert p.is_file()
 acceptance=read(ACCEPT);assert acceptance['status']=='PASS_INDEPENDENT13_FROZEN_TABLE_COUNTS_AND_BILINGUAL_T1H_READBACK' and acceptance['coverage_report_sha256']==sha(COVERAGE)
 coverage=read(COVERAGE);assert coverage['status']=='COMPLETE' and coverage['target_performance_used'] is False
 freeze=R/'data/H/freeze.json';assert sha(freeze)==coverage['provenance']['freeze']['sha256'];f=read(freeze);inputs={str(p):sha(p) for p in [COVERAGE,ACCEPT,AUDIT,CONFIG,freeze]}
 for name,identity in coverage['provenance']['code_dependencies'].items():assert sha(identity['path'])==identity['sha256'];inputs[identity['path']]=identity['sha256']
 for key,identity in f['tables'].items():
  p=R/identity['path'];safe(p);assert sha(p)==acceptance['independent_tables'][key]['table_sha256'];inputs[str(p)]=sha(p)
 roles=coverage['coverage_by_role'];classes=[x for x in coverage['coverage_by_role_method_label'] if x['dataset']=='DFD' and x['role']=='TARGET_TEST'];assert len(roles)==6 and len(classes)==2
 def extract(row):
  n=int(row['candidate_videos']);e=row['processing_eligibility'];out={k:row[k] for k in ['dataset','role','method_id','label'] if k in row};out['candidate_videos']=n
  for name in ['B','common_B_AB']:
   v=e[name];assert 0<=v['numerator']<=v['denominator']==n and abs(v['rate']-v['numerator']/n)<1e-14;out[name]=dict(v)
  assert out['common_B_AB']['numerator']<=out['B']['numerator'];return out
 role_data=[extract(x) for x in roles];class_data=[extract(x) for x in classes];target=next(x for x in role_data if x['dataset']=='DFD');assert [sum(x[k]['numerator'] for x in class_data) for k in ['B','common_B_AB']]==[target[k]['numerator'] for k in ['B','common_B_AB']];assert target['candidate_videos']==3431 and target['B']['numerator']==3320 and target['common_B_AB']['numerator']==3090
 config=read(CONFIG);assert config['artifact_root']==str(R);audit=read(AUDIT);assert audit['status']=='PASS' and audit['CPU_only'] is True and audit['CUDA_initialized'] is False;accepted={x['run_id']:x for x in audit['completed']};assert len(accepted)==len(audit['completed']);curves=[]
 for run,task in config['runs'].items():
  if task['phase']!='development' or task.get('route')!='D':continue
  row=accepted[run];assert row['status']=='PASS_COMPLETED_FROZEN_LAST_MODEL' and row['target_used'] is False and row['independent_DEV_recomputation_max_abs_error']==0;folder=R/'runs'/run
  for name in ['identity.json','resolved_config.json','history.json','complete.json']:
   p=folder/name;safe(p);assert sha(p)==row['core_file_inventory'][name]['sha256'];inputs[str(p)]=sha(p)
  c=read(folder/'complete.json');assert c['target_used'] is False and c['epochs']==task['epochs']==30 and c['selection']=='frozen_last';h=read(folder/'history.json');assert len(h)==30 and [x['epoch'] for x in h]==list(range(1,31))
  for x in h:
   assert x['attempts']==799 and x['draw_count']==102272 and x['success']+x['skip']==799;value=float(x['dev']['macro_auc']);assert np.isfinite(value) and 0<=value<=1;curves.append(dict(run_id=run,backbone=task['backbone'],seed=task['seed'],lr=task['lr'],epoch=x['epoch'],attempts=x['attempts'],successful_updates=x['success'],AMP_skips=x['skip'],source_DEV_method_macro_AUC=value))
 assert len(curves)==300 and len({x['run_id'] for x in curves})==10;safe(OUT.parent);safe(OUT);assert not OUT.exists(),'Preserve existing packet or partial evidence; do not overwrite';assert shutil.disk_usage(R).free>50*1024**3
 OUT.parent.mkdir(exist_ok=True);lock=OUT.parent/'.pre_result_export.lock';safe(lock);fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  info=os.fstat(fd);assert stat.S_ISREG(info.st_mode) and info.st_nlink==1;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists();OUT.mkdir();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'deepfake-v2-pre-result'})
  fig,axes=plt.subplots(1,2,figsize=(12.2,4.6),gridspec_kw={'width_ratios':[2.2,1]},constrained_layout=False);fig.subplots_adjust(left=.065,right=.985,bottom=.22,top=.84,wspace=.3);colors=['#356997','#c26b25']
  for ax,data,title in zip(axes,[role_data,class_data],['a  Processing eligibility by frozen role','b  DFD target class coverage']):
   x=np.arange(len(data));width=.36
   for j,(key,label) in enumerate([('B','B eligible'),('common_B_AB','B/AB common eligible')]):
    values=[100*r[key]['rate'] for r in data];bars=ax.bar(x+(j-.5)*width,values,width,label=label,color=colors[j])
    for bar,row in zip(bars,data):ax.text(bar.get_x()+width/2,5,f"{row[key]['numerator']}/{row['candidate_videos']}",ha='center',va='bottom',rotation=90,color='white',fontsize=8)
   labels=[('DFD\nTARGET TEST' if r['dataset']=='DFD' else 'FF++\n'+r['role'].replace('SOURCE_','').replace('_',' ')) for r in data] if data is role_data else ['Fake' if r['label']==1 else 'Real' for r in data];ax.set_xticks(x,labels,fontsize=8);ax.set_ylim(0,103);ax.set_ylabel('Eligible videos / candidates (%)');ax.set_title(title,loc='left',fontsize=10);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
  handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,fontsize=9);fig.suptitle('Frozen H preprocessing coverage',fontsize=12,y=.98)
  for fmt in ['png','svg']:fig.savefig(OUT/f'H_preprocessing_coverage.{fmt}',dpi=300,metadata={'Date':None} if fmt=='svg' else None)
  plt.close(fig);fig,axes=plt.subplots(1,3,figsize=(12.2,3.8),sharey=True,constrained_layout=True);backbones=['clip_vit_b16','dinov2_vit_b14','efficientnet_b0'];titles=['CLIP ViT-B/16','DINOv2 ViT-B/14','EfficientNet-B0'];all_values=[r['source_DEV_method_macro_AUC'] for r in curves];low=max(0,min(all_values)-.005);high=min(1,max(all_values)+.002)
  for ax,backbone,title in zip(axes,backbones,titles):
   rows=[r for r in curves if r['backbone']==backbone];lrs=sorted({r['lr'] for r in rows});palette=plt.get_cmap('tab10').colors
   for run in dict.fromkeys(r['run_id'] for r in rows):
    part=[r for r in rows if r['run_id']==run];item=part[0];ax.plot([r['epoch'] for r in part],[r['source_DEV_method_macro_AUC'] for r in part],color=palette[lrs.index(item['lr'])],linestyle='-' if item['seed']==1007 else '--',linewidth=1.2,label=f"lr {item['lr']:.0e}, seed {item['seed']}")
   ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch (799 attempted updates each)');ax.set_xlim(1,30);ax.set_ylim(low,high);ax.grid(alpha=.2);ax.legend(fontsize=6.5,frameon=False,loc='lower right')
  axes[0].set_ylabel('Source DEV method-macro AUC');fig.suptitle('All registered D development runs; source DEV only',fontsize=11)
  for fmt in ['png','svg']:fig.savefig(OUT/f'D_source_development_curves.{fmt}',dpi=300,metadata={'Date':None} if fmt=='svg' else None)
  plt.close(fig);(OUT/'coverage_plot_data.json').write_text(json.dumps(dict(roles=role_data,DFD_classes=class_data),indent=2)+'\n');(OUT/'source_development_plot_data.json').write_text(json.dumps(curves,indent=2)+'\n')
  captions='\n\n'.join(['# Pre-result figure captions / 结果前图件图注',
   '**H preprocessing coverage.** Bars show B and joint B/AB processing eligibility among all frozen candidate natural videos, by role and DFD class. Labels are numerator/denominator counts. All SOURCE_UNUSED candidates are administratively excluded from training, calibration and testing even when processing-eligible. TRAIN/DEV exposure uses the common cohort. This is input coverage, without detector accuracy or H GPU-qualification claims. Overlapping failure categories are not added.',
   '**H预处理覆盖率。** 按冻结角色及DFD类别报告B和B/AB共同处理合格率，柱内为自然视频计数的分子/分母。SOURCE_UNUSED即使处理合格仍全部行政排除，不参与训练、校准或测试；TRAIN/DEV实际使用共同队列。图示不是检测准确率或H的GPU资格。不同失败原因不相加。',
   '**D source development curves.** Every registered development run is shown: two seeds per fixed learning-rate candidate. Points are independently accepted source DEV method-macro AUC; shared displayed limits retain every observed point. Each epoch has 799 attempted updates and 102272 sampled inputs. Successes and AMP skips are retained separately in plot data. Source DEV participates in the frozen recipe selection, and does not estimate formal TEST performance or training-seed confidence intervals.',
   '**D源域开发曲线。** 完整显示各登记学习率候选的两个开发种子；每点为已独立验收的源DEV方法宏AUC，三面板使用共同纵轴范围且保留全部观测点。每轮799次更新尝试和102272次输入抽样，成功更新及AMP跳步另存。源DEV参与既定配方选择，不用于推断正式TEST性能或训练种子置信区间。'])+'\n';(OUT/'captions_cn_en.md').write_text(captions)
  assert all(sha(p)==h for p,h in inputs.items()),'Inputs changed while exporting';files={p.name:sha(p) for p in OUT.iterdir() if p.is_file()};record=dict(schema='V2_PRE_RESULT_FIGURE_PACKET_V1',status='TWO_PREPARATION_FIGURES_EXPORTED_RENDER_REVIEW_PENDING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_sha256=inputs,exporter_path=__file__,exporter_sha256=sha(__file__),matplotlib=matplotlib.__version__,artifacts_sha256=files,development_runs=10,development_points=300,H_candidate_videos=8431,source_DEV_performance_included=True,target_performance_read=False,H_GPU_qualification_claimed=False,remaining_figures=['paired_mother_processing_histories','formal_score_ECDF','calibration_variation'],training_or_statistical_protocol_changed=False);(OUT/'complete.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=record['status'],output=str(OUT),files=list(files),bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()))))
 finally:os.close(fd)
if __name__=='__main__':main()
