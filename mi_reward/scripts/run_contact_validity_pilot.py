"""Small MuJoCo contact-validity pilot; not a Factory benchmark."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, mujoco
import torch
from mi_reward.control.tro_image_mi import TROImageMI

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--n',type=int,default=240);args=ap.parse_args();rng=np.random.default_rng(7)
 m=mujoco.MjModel.from_xml_path('assets/custom_task/scene/scene_peg_round_local_servo.xml');r=mujoco.Renderer(m,32,32);mi=TROImageMI(); rows=[]; d=mujoco.MjData(m);mid=int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'ee_target')]);base=d.mocap_pos[mid].copy();
 for i in range(args.n):
  d=mujoco.MjData(m); dx,dy=rng.uniform(-.025,.025,2); dz=rng.uniform(.015,.07); d.mocap_pos[mid]=base+np.array([dx,dy,dz]); mujoco.mj_forward(m,d)
  for _ in range(100): mujoco.mj_step(m,d)
  r.update_scene(d,camera='wrist_cam'); rgb=r.render().astype(np.float32); gray=rgb.mean(-1); r.enable_depth_rendering();r.update_scene(d,camera='wrist_cam');dep=r.render();r.disable_depth_rendering();
  nf=tf=0.;
  for k in range(d.ncon):
   w=np.zeros(6);mujoco.mj_contactForce(m,d,k,w);nf+=abs(w[0]);tf+=np.linalg.norm(w[1:3])
  label=int((abs(dx)+abs(dy)>.018) or tf>1.0 or nf>30)
  ref=gray; score=float(mi(__import__('torch').from_numpy(gray),__import__('torch').from_numpy(ref))[0]); rows.append([*rgb.mean((0,1)).tolist(),float(gray.std()),score,dx,dy,dz,nf,tf,label])
 X=np.asarray(rows); y=X[:,-1].astype(int); idx=rng.permutation(len(y)); tr=idx[:len(y)//2];te=idx[len(y)//2:]; names={'rgb_only':[0,1,2,3],'rgb_mi':[0,1,2,3,4],'rgb_ft':[0,1,2,3,8,9],'full':[0,1,2,3,4,8,9]}; out={}
 def auc(yv,pv):
  order=np.argsort(pv); ranks=np.empty_like(order); ranks[order]=np.arange(len(order)); pos=yv==1; neg=~pos; return float((ranks[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*neg.sum()+1e-9))
 for name,cols in names.items():
  z=torch.tensor(X[tr][:,cols],dtype=torch.float32); t=torch.tensor(y[tr],dtype=torch.float32); w=torch.zeros(z.shape[1],requires_grad=True); b=torch.zeros((),requires_grad=True); opt=torch.optim.Adam([w,b],lr=.05)
  for _ in range(400):
   loss=torch.nn.functional.binary_cross_entropy_with_logits(z@w+b,t); opt.zero_grad();loss.backward();opt.step()
  p=torch.sigmoid(torch.tensor(X[te][:,cols],dtype=torch.float32)@w.detach()+b.detach()).numpy();out[name]={'accuracy':float(np.mean((p>.5)==y[te])),'auroc':auc(y[te],p)}
 report={'benchmark':'custom_mujoco_contact_validity_pilot','n':len(y),'positive_rate':float(y.mean()),'models':out,'note':'Pilot only; Isaac Factory unavailable in current environment.'}
 p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
