"""Temporal contact-validity reward-model pilot on the custom MuJoCo peg scene."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, mujoco, torch
from mi_reward.control.tro_image_mi import TROImageMI

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--n',type=int,default=240);args=ap.parse_args();rng=np.random.default_rng(11); torch.manual_seed(11)
 m=mujoco.MjModel.from_xml_path('assets/custom_task/scene/scene_peg_round_local_servo.xml');mid=int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'ee_target')]);renderer=mujoco.Renderer(m,32,32);mi=TROImageMI();d0=mujoco.MjData(m);mujoco.mj_forward(m,d0);renderer.update_scene(d0,camera='wrist_cam');ref=torch.from_numpy(renderer.render().astype(np.float32).mean(-1)); data=[]
 for _ in range(args.n):
  d=mujoco.MjData(m); base=d.mocap_pos[mid].copy(); dx,dy=rng.uniform(-.025,.025,2); drift=rng.normal(0,.003,(8,2)); seq=[]; final_nf=final_tf=0.
  for t in range(8):
   p=base+np.array([dx+drift[:t+1,0].sum(),dy+drift[:t+1,1].sum(),.06-.004*t]); d.mocap_pos[mid]=p; mujoco.mj_forward(m,d)
   for _ in range(15): mujoco.mj_step(m,d)
   renderer.update_scene(d,camera='wrist_cam'); rgb=renderer.render().astype(np.float32).mean(-1); small=torch.from_numpy(rgb)[::4,::4].reshape(-1).numpy()/255.;
   nf=tf=0.
   for k in range(d.ncon):
    w=np.zeros(6);mujoco.mj_contactForce(m,d,k,w);nf+=abs(w[0]);tf+=np.linalg.norm(w[1:3])
   score=float(mi(torch.from_numpy(rgb),ref)[0]); seq.append(np.r_[small,p-base, nf,tf,score]); final_nf,final_tf=nf,tf
  label=int(abs(dx+drift[:,0].sum())+abs(dy+drift[:,1].sum())>.018 or final_tf>1 or final_nf>30); data.append((np.asarray(seq),label))
 X=np.asarray([x for x,_ in data]); y=np.asarray([z for _,z in data]); perm=rng.permutation(len(y)); tr,te=perm[:len(y)//2],perm[len(y)//2:]
 # per-frame layout: 64 rgb, 3 proprio, 2 F/T, 1 MI
 groups={'rgb_only':list(range(64*8)),'rgb_mi':list(range(64*8))+[69+ i*70 for i in range(8)],'rgb_ft':list(range(64*8))+[67+i*70 for i in range(8)]+[68+i*70 for i in range(8)],'full':list(range(70*8))}
 out={}
 for name,cols in groups.items():
  z=torch.tensor(X.reshape(len(X),-1)[:,cols],dtype=torch.float32); t=torch.tensor(y,dtype=torch.float32); mu=z[tr].mean(0); sd=z[tr].std(0).clamp_min(1e-5); z=(z-mu)/sd; w=torch.zeros(len(cols),requires_grad=True);b=torch.zeros((),requires_grad=True);opt=torch.optim.Adam([w,b],lr=.03)
  for _ in range(300):
   loss=torch.nn.functional.binary_cross_entropy_with_logits(z[tr]@w+b,t[tr]);opt.zero_grad();loss.backward();opt.step()
  p=torch.sigmoid(z[te]@w.detach()+b.detach()).numpy(); order=np.argsort(p); ranks=np.empty_like(order);ranks[order]=np.arange(len(order));pos=y[te]==1;neg=~pos;auc=float((ranks[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*neg.sum()+1e-9));out[name]={'accuracy':float(np.mean((p>.5)==y[te])),'auroc':auc}
 report={'benchmark':'custom_mujoco_contact_validity_reward_pilot','n':len(y),'history':8,'positive_rate':float(y.mean()),'models':out,'note':'Temporal reward-model pilot; not Isaac Factory benchmark.'};p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
