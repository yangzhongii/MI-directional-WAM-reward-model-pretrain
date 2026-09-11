"""Construct paired near-contact hard negatives with hidden physical variation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,mujoco,torch
from mi_reward.control.tro_image_mi import TROImageMI

def contact(m,d,obj,hole):
 nf=tf=0.;pairs=0
 for k in range(d.ncon):
  c=d.contact[k];b1=int(m.geom_bodyid[c.geom1]);b2=int(m.geom_bodyid[c.geom2]);w=np.zeros(6);mujoco.mj_contactForce(m,d,k,w);nf+=abs(w[0]);tf+=np.linalg.norm(w[1:3]);pairs+=int({b1,b2}=={obj,hole})
 return nf,tf,pairs
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--pairs',type=int,default=3);args=ap.parse_args();rng=np.random.default_rng(101)
 m=mujoco.MjModel.from_xml_path('assets/custom_task/scene/scene_peg_round_near_contact.xml');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'task_object');hole=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'insertion_hole');adr=int(m.jnt_qposadr[m.body_jntadr[obj]]);og=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'task_object_geom');r=mujoco.Renderer(m,64,64);mi=TROImageMI();records=[];first=[]
 for pair in range(args.pairs):
  # Keep the same visible initial pose; vary hidden mechanics between paired clips.
  base_q=np.array([.53,.15,.505,1,0,0,0],float); hidden=[{'clearance_mm':.5,'friction':.4,'yaw_deg':0.,'preload_n':0.},{'clearance_mm':.5,'friction':5.0,'yaw_deg':1.5,'preload_n':20.0}]
  ref_img=None
  for cond in hidden:
   d=mujoco.MjData(m);d.qpos[adr:adr+7]=base_q;d.qvel[adr:adr+6]=0.;m.geom_friction[og,0]=cond['friction'];mujoco.mj_forward(m,d);r.update_scene(d,camera='wrist_cam');img0=r.render().copy();first.append(img0);ref_img=img0 if ref_img is None else ref_img;frames=[]
   for t in range(25):
    if t==0 and cond['preload_n']>0:d.qvel[adr+1]=.40
    mujoco.mj_step(m,d);r.update_scene(d,camera='wrist_cam');rgb=r.render().astype(np.uint8);nf,tf,pairs=contact(m,d,obj,hole);score=float(mi(torch.from_numpy(rgb.mean(-1).astype(np.float32)),torch.from_numpy(ref_img.mean(-1).astype(np.float32)))[0]);frames.append({'rgb':rgb,'proprio':d.qpos[:7].astype(np.float32),'ft':np.array([nf,tf],np.float32),'mi':score,'insert_depth':float(d.xpos[obj,2]-d.xpos[hole,2]),'contact_pairs':pairs,'normal_force':nf,'tangential_force':tf})
   valid=sum((f['contact_pairs']>0 or f['insert_depth']<-.015) and f['tangential_force']<5 for f in frames)>=5;label='valid_entry' if valid and max(f['tangential_force'] for f in frames)<5 else ('jam' if max(f['tangential_force'] for f in frames)>=5 else 'false_alignment');records.append({'pair_id':pair,'hidden_condition':cond,'label':label,'frames':frames})
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'frames.npz',rgb=np.asarray([[f['rgb'] for f in x['frames']] for x in records]),proprio=np.asarray([[f['proprio'] for f in x['frames']] for x in records]),ft=np.asarray([[f['ft'] for f in x['frames']] for x in records]),mi=np.asarray([[f['mi'] for f in x['frames']] for x in records]),insert_depth=np.asarray([[f['insert_depth'] for f in x['frames']] for x in records]),contact_pairs=np.asarray([[f['contact_pairs'] for f in x['frames']] for x in records]));first_arr=np.asarray(first);meta={'schema':'contact_hardneg_v5','pairs':args.pairs,'first_frame_pair_mae':float(np.mean(np.abs(first_arr[0::2].astype(float)-first_arr[1::2].astype(float)))),'labels':{k:sum(x['label']==k for x in records) for k in ['valid_entry','jam','false_alignment']},'records':[{k:x[k] for k in ['pair_id','hidden_condition','label']} for x in records]};(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
