"""Collect the redesigned contact-validity dataset contract on the custom peg scene.

This writes trajectories and privileged labels only; it does not train a reward model.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, mujoco, torch
from mi_reward.control.tro_image_mi import TROImageMI

def contact_stats(m,d):
    object_body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'task_object'); socket_body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'insertion_hole'); nf=tf=0.; pairs=[]
    for k in range(d.ncon):
        c=d.contact[k]; b1=int(m.geom_bodyid[c.geom1]); b2=int(m.geom_bodyid[c.geom2]); w=np.zeros(6);mujoco.mj_contactForce(m,d,k,w);nf+=abs(w[0]);tf+=np.linalg.norm(w[1:3]);
        if {b1,b2}=={object_body,socket_body}: pairs.append([b1,b2])
    return float(nf),float(tf),bool(pairs),len(pairs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--trials',type=int,default=96);args=ap.parse_args();rng=np.random.default_rng(23)
    m=mujoco.MjModel.from_xml_path('assets/custom_task/scene/scene_peg_round_local_servo.xml');mid=int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'ee_target')]);obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'task_object');hole=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'insertion_hole');renderer=mujoco.Renderer(m,64,64);mi=TROImageMI();
    # Fixed pre-contact goal/reference image for every trial.
    ref_data=mujoco.MjData(m);base=ref_data.mocap_pos[mid].copy();ref_data.mocap_pos[mid]=base+np.array([.05,.31,.08]);mujoco.mj_forward(m,ref_data);renderer.update_scene(ref_data,camera='wrist_cam');ref=torch.from_numpy(renderer.render().astype(np.float32).mean(-1));
    records=[]; conditions=['successful_entry','lateral_jam','false_alignment']
    for trial in range(args.trials):
        condition=conditions[trial%len(conditions)]; d=mujoco.MjData(m); seq=[]; base_shift=np.array([.05,.31,.08]);
        if condition=='lateral_jam': offset=np.array([rng.uniform(.014,.025),rng.uniform(.014,.025),0.])
        elif condition=='false_alignment': offset=np.array([rng.uniform(.004,.009),rng.uniform(.004,.009),.01])
        else: offset=np.array([rng.uniform(-.002,.002),rng.uniform(-.002,.002),0.])
        for t in range(20):
            alpha=min(1.,t/10); shift=base_shift*alpha+offset*(alpha if t<14 else 1.); d.mocap_pos[mid]=base+shift+np.array([0,0,-.002*max(0,t-10)]);mujoco.mj_forward(m,d)
            for _ in range(10):mujoco.mj_step(m,d)
            renderer.update_scene(d,camera='wrist_cam');rgb=renderer.render().astype(np.float32); gray=rgb.mean(-1); nf,tf,contact,n_pairs=contact_stats(m,d);score=float(mi(torch.from_numpy(gray),ref)[0]);depth=float(d.xpos[obj,2]-d.xpos[hole,2]);
            seq.append({'rgb':rgb[::4,::4].astype(np.uint8),'proprio':np.r_[d.mocap_pos[mid],d.mocap_quat[mid]].astype(np.float32),'ft':np.array([nf,tf],np.float32),'mi':score,'insert_depth':depth,'contact_valid':contact,'contact_pairs':n_pairs,'normal_force':nf,'tangential_force':tf})
        # Labels are derived from contact/depth/force/recoverability, not offsets.
        valid_frames=[x for x in seq if x['contact_valid'] and x['insert_depth']<-.005 and x['tangential_force']<5.]; peak_tf=max(x['tangential_force'] for x in seq); label='valid_entry' if len(valid_frames)>=3 and peak_tf<5 else ('jam' if peak_tf>=5 or condition=='lateral_jam' else 'false_alignment')
        records.append({'trial_id':trial,'condition':condition,'label':label,'frames':seq})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'frames.npz',rgb=np.asarray([[f['rgb'] for f in r['frames']] for r in records]),proprio=np.asarray([[f['proprio'] for f in r['frames']] for r in records]),ft=np.asarray([[f['ft'] for f in r['frames']] for r in records]),mi=np.asarray([[f['mi'] for f in r['frames']] for r in records]),insert_depth=np.asarray([[f['insert_depth'] for f in r['frames']] for r in records]),contact_valid=np.asarray([[f['contact_valid'] for f in r['frames']] for r in records]),normal_force=np.asarray([[f['normal_force'] for f in r['frames']] for r in records]),tangential_force=np.asarray([[f['tangential_force'] for f in r['frames']] for r in records]))
    meta={'schema':'contact_validity_v2','reference':'fixed_precontact_goal_image','trial_split':'trial/condition grouped; no frame random split','num_trials':len(records),'labels':{'valid_entry':sum(r['label']=='valid_entry' for r in records),'jam':sum(r['label']=='jam' for r in records),'false_alignment':sum(r['label']=='false_alignment' for r in records)},'trials':[{k:r[k] for k in ('trial_id','condition','label')} for r in records]};(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
