"""Collect near-contact peg clips with privileged contact-validity labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,mujoco,torch
from mi_reward.control.tro_image_mi import TROImageMI

def stats(m,d,obj,hole):
 nf=tf=0.;pairs=0
 for k in range(d.ncon):
  c=d.contact[k];b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2]);w=np.zeros(6);mujoco.mj_contactForce(m,d,k,w);nf+=abs(w[0]);tf+=np.linalg.norm(w[1:3]);pairs+=int({b1,b2}=={obj,hole})
 return nf,tf,pairs
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--trials',type=int,default=30);args=ap.parse_args();rng=np.random.default_rng(31)
 m=mujoco.MjModel.from_xml_path('assets/custom_task/scene/scene_peg_round_near_contact.xml');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'task_object');hole=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'insertion_hole');adr=int(m.jnt_qposadr[m.body_jntadr[obj]]);r=mujoco.Renderer(m,64,64);mi=TROImageMI();d0=mujoco.MjData(m);mujoco.mj_forward(m,d0);r.update_scene(d0,camera='wrist_cam');ref=torch.from_numpy(r.render().astype(np.float32).mean(-1));records=[]
 for i in range(args.trials):
  kind=['success','jam','false_alignment'][i%3];d=mujoco.MjData(m);q=np.array([.53,.15,.555,1,0,0,0],float)
  if kind=='jam':q[:2]+=rng.uniform(-.012,.012,2)
  if kind=='success':q[2]=.505
  if kind=='false_alignment':q[2]=.60
  d.qpos[adr:adr+7]=q;mujoco.mj_forward(m,d);frames=[]
  for t in range(25):
   mujoco.mj_step(m,d);r.update_scene(d,camera='wrist_cam');rgb=r.render().astype(np.uint8);nf,tf,pairs=stats(m,d,obj,hole);score=float(mi(torch.from_numpy(rgb.mean(-1).astype(np.float32)),ref)[0]);frames.append({'rgb':rgb[::4,::4],'proprio':np.array(d.qpos[:7],np.float32),'ft':np.array([nf,tf],np.float32),'mi':score,'insert_depth':float(d.xpos[obj,2]-d.xpos[hole,2]),'contact_pairs':pairs,'normal_force':nf,'tangential_force':tf})
  valid=sum((f['contact_pairs']>0 or f['insert_depth']<-.015) and f['insert_depth']<.02 and f['tangential_force']<5 for f in frames)>=5;label='valid_entry' if valid and kind=='success' else ('jam' if max(f['tangential_force'] for f in frames)>5 or kind=='jam' else 'false_alignment');records.append({'trial_id':i,'condition':kind,'label':label,'frames':frames})
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'frames.npz',rgb=np.asarray([[f['rgb'] for f in x['frames']] for x in records]),proprio=np.asarray([[f['proprio'] for f in x['frames']] for x in records]),ft=np.asarray([[f['ft'] for f in x['frames']] for x in records]),mi=np.asarray([[f['mi'] for f in x['frames']] for x in records]),insert_depth=np.asarray([[f['insert_depth'] for f in x['frames']] for x in records]),contact_pairs=np.asarray([[f['contact_pairs'] for f in x['frames']] for x in records]));meta={'schema':'near_contact_reward_v1','split':'trial-level','labels':{k:sum(x['label']==k for x in records) for k in ['valid_entry','jam','false_alignment']},'trials':[{k:x[k] for k in ['trial_id','condition','label']} for x in records]};(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
