"""Render fixed source-only mother/B/A/AB histories without reprocessing pixels."""
from pathlib import Path
import os,json,hashlib,datetime,shutil,fcntl,stat
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reliability_v2.geometry_h import rgb_hash,read_rgb
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
SELECT=R/'reports/paper_paired_history_illustration_selection.json'
COVERAGE=R/'reports/H_frozen_coverage_repaired_20261024T0320.json'
ACCEPT=R/'reports/H_coverage_T1_table_independent_acceptance_20261024T0320.json'
OUT=R/'manuscript_figures/paired_histories'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cv2.setNumThreads(2)
 selection=read(SELECT);assert selection['status']=='FIXED_SOURCE_COMMON_TRAIN_ILLUSTRATION_INPUTS_ONLY' and selection['source_only'] and selection['target_or_detector_scores_read'] is False
 freeze=R/'data/H/freeze.json';assert sha(freeze)==selection['freeze_sha256'];accept=read(ACCEPT);assert sha(COVERAGE)==accept['coverage_report_sha256'];coverage=read(COVERAGE);expected={x['path']:x['sha256'] for x in coverage['provenance']['processed_manifests']};inputs={str(p):sha(p) for p in [SELECT,COVERAGE,ACCEPT,freeze]};records=[];images=[]
 for key,identity in selection['table_identities'].items():p=R/identity['path'];assert sha(p)==identity['sha256'];inputs[str(p)]=sha(p)
 assert len(selection['selections'])==5
 for chosen in selection['selections']:
  b=chosen['B'];assert b['domain']=='source' and b['role']=='TRAIN';manifest=Path(b['image_path']).parent.parent/'manifest.json';assert sha(manifest)==expected[str(manifest)];inputs[str(manifest)]=sha(manifest);m=read(manifest);assert m['video']['video_uid']==b['video_uid'] and m['common_B_AB'] and m['video']['role']=='TRAIN' and m['video']['domain']=='source'
  frames=[x for x in m['frames'] if x['mother_index']==b['mother_frame_index'] and x['mother_rgb_sha256']==b['mother_rgb_sha256']];assert len(frames)==1;frame=frames[0]
  p=Path(frame['mother_path']);assert sha(p)==frame['mother_byte_sha256'];mother=read_rgb(p);assert rgb_hash(mother)==frame['mother_rgb_sha256'];inputs[str(p)]=sha(p);row=[mother];masks={};derived={}
  for version in ['B','A','AB']:
   v=frame[version];assert v['ok'];p=Path(v['path']);maskp=Path(v['mask_path']);assert sha(p)==v['byte_sha256'] and sha(maskp)==v['mask_byte_sha256'];rgb=read_rgb(p);mask=cv2.imread(str(maskp),cv2.IMREAD_GRAYSCALE);assert rgb_hash(rgb)==v['rgb_sha256'] and mask is not None and mask.shape==rgb.shape[:2] and set(np.unique(mask))<={0,255};padding=1-np.count_nonzero(mask)/mask.size;assert abs(padding-v['padding_ratio'])<1e-12;inputs[str(p)]=sha(p);inputs[str(maskp)]=sha(maskp);row.append(rgb);masks[version]=mask;derived[version]=dict(padding_ratio=float(padding),image_shape=list(rgb.shape),image_byte_sha256=v['byte_sha256'],image_rgb_sha256=v['rgb_sha256'],mask_byte_sha256=v['mask_byte_sha256'])
  assert frame['A']['input_rgb_sha256']==frame['B']['input_rgb_sha256']==frame['mother_rgb_sha256'] and frame['AB']['input_rgb_sha256']==frame['A']['rgb_sha256'] and frame['AB']['parent_byte_sha256']==frame['A']['byte_sha256']
  for version in ['B','AB']:assert chosen[version]['image_path']==frame[version]['path'] and chosen[version]['content_sha256']==frame[version]['byte_sha256'] and chosen[version]['rgb_sha256']==frame[version]['rgb_sha256']
  row.append(masks['AB']);images.append(row);records.append(dict(method_id=chosen['method_id'],mother_id=chosen['mother_id'],video_uid=b['video_uid'],frame_index=frame['mother_index'],source_manifest=str(manifest),mother_shape=list(mother.shape),derived=derived,original_frame_metadata=frame))
 assert not OUT.exists() and OUT.resolve().is_relative_to(R) and not OUT.parent.is_symlink() and shutil.disk_usage(R).free>50*1024**3;OUT.parent.mkdir(exist_ok=True);lock=OUT.parent/'.paired_histories.lock';fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  info=os.fstat(fd);assert stat.S_ISREG(info.st_mode) and info.st_nlink==1;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not OUT.exists();OUT.mkdir();plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'deepfake-v2-paired-histories'});fig,axes=plt.subplots(5,5,figsize=(12.4,11.2));fig.subplots_adjust(left=.085,right=.99,top=.92,bottom=.035,wspace=.1,hspace=.3);headers=['Mother RGB','B from mother','A intermediate','AB from A','AB validity mask']
  for i,(row,record) in enumerate(zip(images,records)):
   for j,img in enumerate(row):
    ax=axes[i,j];ax.imshow(img,cmap='gray' if j==4 else None,vmin=0 if j==4 else None,vmax=255 if j==4 else None,interpolation='nearest');ax.set_axis_off()
    if i==0:ax.set_title(headers[j],fontsize=10,pad=8)
    note='frame '+str(record['frame_index']) if j==0 else 'white = valid' if j==4 else 'padding '+format(record['derived'][['B','A','AB'][j-1]]['padding_ratio'],'.1%');ax.text(.5,-.055,note,transform=ax.transAxes,ha='center',va='top',fontsize=8)
   pos=axes[i,0].get_position();fig.text(.012,(pos.y0+pos.y1)/2,record['method_id'],rotation=90,ha='center',va='center',fontsize=10)
  fig.suptitle('Fixed source TRAIN mothers and their processing histories',fontsize=12,y=.985)
  for fmt in ['png','svg']:fig.savefig(OUT/f'paired_mother_processing_histories.{fmt}',dpi=300,metadata={'Date':None} if fmt=='svg' else None)
  plt.close(fig);(OUT/'paired_history_plot_data.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n');captions='\n\n'.join(['# Paired processing-history illustration / 同母帧处理示意',
  'The five rows cover all H source method labels and were fixed by lexicographic (video_uid, mother_frame_index, mother_id) selection within frozen common TRAIN, without consulting deepfake model-performance scores or adding a new geometry-confidence ranking. Saved SCRFD geometry metadata are retained and frozen common eligibility is unchanged. Columns retain the full saved mother, direct B crop, A intermediate, AB crop after A redetection, and AB validity mask (white valid, black padding). Image aspect ratios are retained; saved pixels are displayed with nearest interpolation, without new face detection, alignment, cropping or retouching. Padding fractions are recomputed from the saved masks. A/B/AB byte, RGB, mask and parent-chain identities are verified. Examples may share source people or ancestry and are illustrative, not independent evidence of image quality, manipulated-face identity, or model accuracy. No target TEST performance or H GPU qualification is claimed.',
  '五行覆盖H的全部源方法标签，按冻结共同TRAIN中的(video_uid, mother_frame_index, mother_id)字典序预选，不使用deepfake模型性能分数或新增几何置信度排序来挑图；保留既有SCRFD几何元数据，不改变冻结共同资格。各列为完整已保存母帧、直接B裁剪、A中间图、A后重检测所得AB裁剪及AB有效像素mask（白为有效、黑为padding）。保留纵横比，仅以nearest方式显示保存像素，不重新检测、对齐、裁剪或修饰；padding比例由已保存mask复算，并验证字节/RGB/mask及父图身份链。示例可能共享人物或祖先，仅作流程示意，不是独立的质量、操纵身份或模型准确率证据，也不表示目标测试性能或H的GPU资格。'])+'\n';(OUT/'captions_cn_en.md').write_text(captions);assert all(sha(p)==v for p,v in inputs.items());files={p.name:sha(p) for p in OUT.iterdir()};record=dict(schema='V2_PAIRED_HISTORY_FIGURE_V1',status='RENDERED_NUMERIC_AND_VISUAL_REVIEW_PENDING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_sha256=sha(SELECT),input_sha256=inputs,exporter_path=__file__,exporter_sha256=sha(__file__),artifacts_sha256=files,source_methods=[r['method_id'] for r in records],source_mothers=5,image_pixels_reprocessed=False,detector_or_model_run=False,target_or_deepfake_model_scores_read=False,saved_face_geometry_metadata_read=True,additional_geometry_confidence_ranking_for_illustration=False,source_masks_recomputed=15);(OUT/'complete.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=record['status'],output=str(OUT),bytes=sum(p.stat().st_size for p in OUT.iterdir()))))
 finally:os.close(fd)
if __name__=='__main__':main()
