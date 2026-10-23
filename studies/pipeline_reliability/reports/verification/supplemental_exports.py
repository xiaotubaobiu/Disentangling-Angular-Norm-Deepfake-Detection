"""Supplemental scientific exports from accepted, immutable saved scores/tables."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,hashlib,shutil,datetime,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=pathlib.Path('/home/z/project/deepfake_detector');R=(P/'artifacts').resolve();D=P/'reports/final_report'
accept=json.loads((P/'reports/final_independent_acceptance.json').read_text());assert accept['status']=='INDEPENDENT_FINAL_ACCEPTANCE_PASS'
assert json.loads((R/'execution_status.json').read_text())['status']=='COMPLETE'
assert not D.exists(),'Preserve existing export directory; inspect it before a new version'
D.mkdir()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
source_hashes={}
for p in sorted((R/'report').iterdir()):
 if p.is_file() and p.suffix in ['.csv','.json','.png']:
  shutil.copy2(p,D/p.name);source_hashes[p.name]=sha(p);assert sha(D/p.name)==source_hashes[p.name]
c=json.loads((P/'configs/v1.1.json').read_text());seeds=c['seeds'];frame=pd.read_csv(D/'metrics_by_seed.csv');assert len(frame)==180
pairs=pd.read_csv(D/'paired_flow_differences.csv');video=frame[frame.unit.eq('video')];assert len(video)==120
contrast_names=['delta_raw','delta_dir','delta_norm_L','delta_norm_P','interaction'];titles=['Raw: RP - RL','Direction: DP - DL','Norm: DL - RL','Norm: DP - RP','Interaction']
def contrasts(v):
 rl,rp,dl,dp=[v[k] for k in ['RL','RP','DL','DP']]
 return [rp-rl,dp-dl,dl-rl,dp-rp,dp-dl-rp+rl]
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
rows=[];files=[]
colors=plt.get_cmap('tab10').colors[:5];offsets=np.linspace(-.16,.16,5)
for backbone in c['backbones']:
 fig,axes=plt.subplots(2,5,figsize=(17,7),constrained_layout=True)
 for mi,metric in enumerate(['method_macro_AUC','FPR']):
  for di,domain in enumerate(['ff','cdf']):
   table=video[video.backbone.eq(backbone)&video.domain.eq(domain)].pivot(index='seed',columns='flow',values=metric).loc[seeds]
   values=np.array([contrasts(table.loc[s]) for s in seeds])
   if mi==0:
    saved=pairs[pairs.backbone.eq(backbone)&pairs.domain.eq(domain)].set_index('seed').loc[seeds]
    assert np.allclose(values,saved[contrast_names],atol=1e-12,rtol=0)
   for ci,name in enumerate(contrast_names):
    for si,seed in enumerate(seeds):
     rows.append({'backbone':backbone,'seed':seed,'domain':domain,'metric':metric,'contrast':name,'delta':float(values[si,ci]),'delta_percentage_points':float(100*values[si,ci])})
     axes[mi,ci].scatter(di+offsets[si],100*values[si,ci],color=colors[si],s=24,label=f'seed {seed}' if di==0 else None,zorder=3)
    vv=100*values[:,ci];axes[mi,ci].errorbar(di,vv.mean(),yerr=vv.std(ddof=1),fmt='D',color='black',markersize=4,capsize=3,zorder=4)
  for ci,ax in enumerate(axes[mi]):
   ax.axhline(0,color='grey',linestyle='--',linewidth=.8);ax.set_xticks([0,1],['FF test','CDF test']);ax.set_xlim(-.4,1.4);ax.set_title(titles[ci]);ax.grid(axis='y',alpha=.2)
   if ci==0:ax.set_ylabel(('Method-macro AUC' if mi==0 else 'FPR')+' difference (pp)')
 axes[0,0].legend(fontsize=7)
 fig.suptitle(backbone+' | all five paired seeds; black diamond = mean, bar = seed SD\nFPR contrasts use each flow\'s own frozen source threshold; higher FPR is worse',fontsize=11)
 for ext in ['png','pdf']:
  name=backbone+'_paired_flow_differences.'+ext;fig.savefig(D/name,dpi=160);files.append(name)
 plt.close(fig)
pd.DataFrame(rows).to_csv(D/'paired_metric_differences_all_seeds.csv',index=False);assert len(rows)==300
coverage=[]
flows=['video_RL','video_RP','video_DL','video_DP','image_raw','image_direction']
for backbone in c['backbones']:
 fig,axes=plt.subplots(6,5,figsize=(17,19),constrained_layout=True)
 for si,seed in enumerate(seeds):
  folder=R/'analysis'/f'{backbone}_seed{seed}'/'scores'
  for fi,flow in enumerate(flows):
   cols=['label','score','weight','threshold'];cal=pd.read_parquet(folder/'calibration'/(flow+'.parquet'),columns=cols);cal=cal[cal.label.eq(0)];t=float(cal.threshold.iloc[0]);mean=float(np.average(cal.score,weights=cal.weight));sd=float(np.sqrt(np.average((cal.score-mean)**2,weights=cal.weight)));effective_sd=sd if sd>0 else 1.
   ax=axes[fi,si]
   for domain,label in [('calibration','Source cal'),('ff','FF test'),('cdf','CDF test')]:
    data=cal if domain=='calibration' else pd.read_parquet(folder/domain/(flow+'.parquet'),columns=cols);data=data[data.label.eq(0)];z=(data.score.to_numpy()-t)/effective_sd;w=data.weight.to_numpy();order=np.argsort(z,kind='stable');ax.step(z[order],np.cumsum(w[order])/w.sum(),where='post',label=label,linewidth=1)
    fpr=float(np.average(data.score>=t,weights=w));coverage.append({'backbone':backbone,'seed':seed,'flow':flow,'domain':domain,'real_rows':len(data),'source_real_SD':sd,'source_SD_fallback_to_1':sd==0,'weighted_mass_below_minus4':float(np.average(z < -4,weights=w)),'weighted_mass_above_plus4':float(np.average(z > 4,weights=w)),'frozen_threshold_FPR':fpr,'relative_score_min':float(z.min()),'relative_score_max':float(z.max())})
   ax.axvline(0,color='black',linewidth=.8);ax.set_ylim(0,1);ax.set_title(flow+' seed '+str(seed),fontsize=8);ax.set_xlabel('(score - source threshold) / source real SD',fontsize=6);ax.legend(fontsize=5)
   if si==0:ax.set_ylabel('Weighted empirical CDF',fontsize=7)
 fig.suptitle(backbone+' | Real-score distributions over the full observed range\nAll real samples retained. Decisions use inclusive score >= frozen source threshold.',fontsize=11)
 for ext in ['png','pdf']:
  name=backbone+'_real_scores_relative_full_range_ECDF.'+ext;fig.savefig(D/name,dpi=130);files.append(name)
 plt.close(fig)
pd.DataFrame(coverage).to_csv(D/'relative_score_plot_coverage.csv',index=False);assert len(coverage)==270
record={'status':'SUPPLEMENTAL_EXPORTS_COMPLETE','checked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'accepted_primary_source_table_and_figure_sha256':source_hashes,'supplemental_figures':files,'paired_point_rows':300,'relative_distribution_coverage_rows':270,'all_five_seed_points_retained_per_backbone':True,'no_model_or_threshold_or_protocol_changes':True,'original_six_figures_preserved_byte_for_byte':True,'export_helper_sha256':os.environ.get('DF40_EXPORT_HELPER_SHA256'),'outputs':{p.name:sha(p) for p in sorted(D.iterdir()) if p.is_file()}}
(D/'supplemental_export_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'canonical_figures':6,'supplemental_PNG_figures':6,'paired_rows':300,'coverage_rows':270,'maximum_original_histogram_unshown_weighted_mass':max(x['weighted_mass_below_minus4']+x['weighted_mass_above_plus4'] for x in coverage)}))
