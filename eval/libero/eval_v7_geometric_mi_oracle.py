"""Pipeline-v7 G1: actual-next Dame MI on fixed MuJoCo landmark tokens only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
import h5py, numpy as np, torch, torch.nn.functional as F
from mi_reward.control.mi_action_field import fit_fixed_normalization, normalize_tokens
from mi_reward.scoring.dame_soft_histogram import DameSoftHistogramMI
from mi_reward.data.libero_privileged import resolve_libero_task
from collect_v7_geometric_correspondences import _identity_table, _points

def _args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--collection-dir',type=Path,required=True); p.add_argument('--correspondence-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); return p.parse_args()
def _metric(a,b):
 x,y=np.asarray(a),np.asarray(b); v=float(((y-y.mean())**2).sum()); nz=np.abs(y)>1e-9
 return {'r2':None if v<=1e-12 else float(1-((x-y)**2).sum()/v),'spearman':None if not np.std(x) or not np.std(y) else float(np.corrcoef(np.argsort(np.argsort(x)),np.argsort(np.argsort(y)))[0,1]),'sign_accuracy':None if not np.any(nz) else float(np.mean(np.sign(x[nz])==np.sign(y[nz])))}
def _tokens(points, physical):
 task=sorted(points['task_object_points'],key=lambda x:x['identity']); goal_site=next(x for x in points['goal_object_points'] if x['kind']=='site')
 world=np.asarray([x['world_xyz'] for x in task]); eef=np.asarray(physical['eef_pos']); goal=np.asarray(goal_site['world_xyz'])
 return torch.tensor(np.concatenate((world-eef,world-goal),axis=1),dtype=torch.float32)
def _mi(z,g,norm,est):
 z,_=normalize_tokens(z,norm); g,_=normalize_tokens(g,norm); return est(z,g)
def main():
 args=_args()
 if args.output_dir.exists(): raise FileExistsError(args.output_dir)
 rows=[json.loads(x) for x in (args.collection_dir/'manifest.jsonl').read_text().splitlines() if x]
 corr=json.loads((args.correspondence_dir/'correspondences.json').read_text()); by_anchor={x['anchor_id']:x for x in corr['anchors']}
 if len(rows)!=20 or set(by_anchor)!=set(x['anchor_id'] for x in rows): raise RuntimeError('Requires exact frozen 20-anchor correspondence schema')
 # Reconstruct the independent reference with the same fixed MuJoCo identities.
 root=Path.cwd(); demo,bddl,_=resolve_libero_task(root,rows[0]['suite'],int(rows[0]['task_id'])); ref=rows[0]['reference']
 from libero.libero.envs import OffScreenRenderEnv
 env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_names=['agentview'],camera_heights=128,camera_widths=128)
 try:
  with h5py.File(demo,'r') as f: state=np.asarray(f['data'][ref['demo_name']]['states'][int(ref['frame_index'])],dtype=np.float64)
  env.reset(); env.set_init_state(state); core=env.env; _,task_name,goal_name=__import__('mi_reward.data.libero_privileged',fromlist=['goal_objects']).goal_objects(core.parsed_problem['goal_state'])
  tr,tt=_identity_table(core,task_name,'task_object'); gr,gt=_identity_table(core,goal_name,'goal_object')
  ref_points={'task_object_points':_points(core,tr,tt),'goal_object_points':_points(core,gr,gt)}; reference=_tokens(ref_points,ref['physical'])
 finally: env.close()
 est=DameSoftHistogramMI(num_bins=8,spline_order=3,normalization='none',channel_mode='channelwise')
 all_score=[]; all_progress=[]; cosines=[]; cross=[]; descent=[]; anchors=[]
 for row in rows:
  candidates={x['candidate_id']:x for x in by_anchor[row['anchor_id']]['candidates']}; center=candidates['center']; z0=_tokens(center, next(x['physical_after'] for x in row['candidates'] if x['candidate_id']=='center')); norm=fit_fixed_normalization(reference,[z0],fit_anchor_ids=(row['anchor_id'],)); m0=_mi(z0,reference,norm,est)
  manifest={x['candidate_id']:x for x in row['candidates']}
  def grad(label):
   eef=torch.stack([(torch.tensor(manifest[f'{label}_plus_{i}']['physical_after']['eef_pos'])-torch.tensor(manifest['center']['physical_after']['eef_pos'])-(torch.tensor(manifest[f'{label}_minus_{i}']['physical_after']['eef_pos'])-torch.tensor(manifest['center']['physical_after']['eef_pos'])))/2 for i in range(3)],dim=1).float()
   vals=[]
   for i in range(3):
    for sign in ('plus','minus'):
     c=candidates[f'{label}_{sign}_{i}']; p=manifest[f'{label}_{sign}_{i}']['physical_after']; vals.append(_mi(_tokens(c,p),reference,norm,est))
   return torch.linalg.pinv(eef.T)@((torch.stack(vals[::2])-torch.stack(vals[1::2]))/2)
  gp,gs=grad('primary'),grad('secondary'); physical=-torch.linalg.pinv(torch.stack([(torch.tensor(manifest[f'primary_plus_{i}']['physical_after']['eef_pos'])-torch.tensor(manifest['center']['physical_after']['eef_pos'])-(torch.tensor(manifest[f'primary_minus_{i}']['physical_after']['eef_pos'])-torch.tensor(manifest['center']['physical_after']['eef_pos'])))/2 for i in range(3)],dim=1).float().T)@torch.tensor([(manifest[f'primary_plus_{i}']['physical_after']['eef_object_distance']-manifest[f'primary_minus_{i}']['physical_after']['eef_object_distance'])/2 for i in range(3)])
  co=float(F.cosine_similarity(gp[None],physical[None]).item()); ce=float(F.cosine_similarity(gp[None],gs[None]).item()); scores=[]; progress=[]
  for name,c in candidates.items():
   if name.startswith('heldout_'):
    p=manifest[name]['physical_after']; scores.append(float(_mi(_tokens(c,p),reference,norm,est)-m0)); progress.append(-(p['eef_object_distance']-manifest['center']['physical_after']['eef_object_distance']))
  all_score+=scores;all_progress+=progress;cosines.append(co);cross.append(ce);descent.append(co>0);anchors.append({'anchor_id':row['anchor_id'],'source_demo':row['source_demo'],'heldout':_metric(scores,progress),'physical_cosine':co,'cross_epsilon_cosine':ce})
  print(f"ANCHOR {row['anchor_id']} physical={co:.6f} cross_epsilon={ce:.6f}",flush=True)
 held=_metric(all_score,all_progress); passed=held['r2'] is not None and held['r2']>=.5 and held['spearman']>=.6 and held['sign_accuracy']>=.8 and float(np.median(cosines))>=.7 and float(np.median(cross))>=.9
 report={'protocol':'v7_g1_actual_next_geometric_mi_v1','branch':'G1_geometric_actual_next_mi','privileged_input':True,'deployable':False,'actual_or_predicted':'actual','mi_normalization':'per-anchor fixed reference plus center tokens','heldout_action_count':len(all_score),'heldout':held,'physical_cosine_median':float(np.median(cosines)),'distance_descent_fraction':float(np.mean(descent)),'cross_epsilon_cosine_median':float(np.median(cross)),'per_anchor':anchors,'status':'PASS' if passed else 'NO_GO'}
 args.output_dir.mkdir(parents=True);(args.output_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','heldout','physical_cosine_median','cross_epsilon_cosine_median')},indent=2),flush=True)
if __name__=='__main__': main()
