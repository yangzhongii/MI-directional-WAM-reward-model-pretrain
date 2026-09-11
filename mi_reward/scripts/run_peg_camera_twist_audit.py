"""Calibrate MuJoCo wrist-camera pose perturbations against point interaction Lx."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, mujoco, torch
from mi_reward.control.tro_image_interaction import point_interaction_matrix

def quat_mul(a,b):
    w,x,y,z=a; W,X,Y,Z=b
    return np.array([w*W-x*X+y*Z-z*Y,w*X+x*W+y*Z-z*X,w*Y-x*Z+y*W-z*Y,w*Z+x*Y-y*X+z*W])

def project(m,d,cid,p):
    pos=d.cam_xpos[cid]; R=d.cam_xmat[cid].reshape(3,3); q=R@(p-pos); z=-q[2]
    f=0.5*128/np.tan(np.deg2rad(float(m.cam_fovy[cid]))/2); return np.array([63.5+f*q[0]/z,63.5+f*q[1]/z]),z

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='assets/custom_task/scene/scene_peg_round_local_servo.xml'); ap.add_argument('--output',required=True); args=ap.parse_args()
    m=mujoco.MjModel.from_xml_path(args.model); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    mid=int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'ee_target')]); cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'wrist_cam'); hid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'insertion_hole'); point=np.array(d.xpos[hid],copy=True)
    base_pos=np.array(d.mocap_pos[mid],copy=True); base_q=np.array(d.mocap_quat[mid],copy=True); mujoco.mj_forward(m,d); uv0,z0=project(m,d,cid,point); f=0.5*128/np.tan(np.deg2rad(float(m.cam_fovy[cid]))/2)
    rows=[]
    for j in range(6):
        h=1e-4; vals=[]
        for s in (-1,1):
            d.mocap_pos[mid]=base_pos; d.mocap_quat[mid]=base_q
            if j<3: d.mocap_pos[mid,j]+=s*h
            else:
                a=np.zeros(3);a[j-3]=1; d.mocap_quat[mid]=quat_mul(base_q,np.r_[np.cos(h/2),a*np.sin(s*h/2)])
            mujoco.mj_forward(m,d); uv,z=project(m,d,cid,point); vals.append(uv)
        obs=(vals[1]-vals[0])/(2*h)
        depth=torch.tensor(float(z0)); L=point_interaction_matrix(1,1,f,f,63.5,63.5,depth[None,None])[0,0]
        rows.append({'dof':j,'observed_pixel_derivative':obs.tolist(),'Lx_rows':L.tolist()})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'stage':'camera_twist_audit','point_camera_depth':float(z0),'point_pixel':uv0.tolist(),'rows':rows},indent=2)+'\n');print(json.dumps({'point_camera_depth':float(z0),'point_pixel':uv0.tolist(),'rows':rows},indent=2))
if __name__=='__main__':main()
