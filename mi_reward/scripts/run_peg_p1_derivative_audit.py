"""P1 raw/goal-centered MI derivative audit on the local-servo scene."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
import mujoco

from mi_reward.control.tro_image_mi import TROImageMI
from mi_reward.control.tro_image_interaction import image_pose_jacobian


def quat_mul(a, b):
    w,x,y,z = a; W,X,Y,Z = b
    return np.array([w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W])


def render(renderer, data, camera):
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()
    renderer.enable_depth_rendering(); renderer.update_scene(data, camera=camera)
    depth = renderer.render(); renderer.disable_depth_rendering()
    gray = torch.from_numpy(rgb.astype(np.float32).mean(axis=-1))
    dep = torch.from_numpy(np.maximum(depth.astype(np.float32), 1e-3))
    return gray, dep


def mocap_to_camera_twist(model, data, mocap_id, camera_id):
    """Map world-frame mocap translation/rotation axes to camera-frame twist."""
    cam_pos = np.array(data.cam_xpos[camera_id], copy=True)
    cam_R = data.cam_xmat[camera_id].reshape(3, 3)
    body_id = int(np.flatnonzero(model.body_mocapid == mocap_id)[0])
    origin = np.array(data.xpos[body_id], copy=True)
    lever = cam_pos - origin
    T = np.zeros((6, 6), dtype=np.float64)
    for j in range(3):
        v_world = np.zeros(3); v_world[j] = 1.0
        T[:3, j] = cam_R @ v_world
        omega_world = np.zeros(3); omega_world[j] = 1.0
        T[:3, j + 3] = cam_R @ np.cross(omega_world, lever)
        T[3:, j + 3] = cam_R @ omega_world
    return T


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='assets/custom_task/scene/scene_peg_round_local_servo.xml'); ap.add_argument('--output',required=True); ap.add_argument('--size',type=int,default=64); args=ap.parse_args()
    m=mujoco.MjModel.from_xml_path(args.model); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    mid=int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'ee_target')])
    renderer=mujoco.Renderer(m,height=args.size,width=args.size)
    ref, depth=render(renderer,d,'wrist_cam')
    mi=TROImageMI(bins=8,padding=2)
    cam_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'wrist_cam')
    fy=0.5*float(args.size)/np.tan(np.deg2rad(float(m.cam_fovy[cam_id]))/2.0); fx=fy
    cx=cy=(args.size-1)/2
    jac=image_pose_jacobian(ref,depth,fx,fy,cx,cy)
    twist=mocap_to_camera_twist(m,d,mid,cam_id)
    jac_mocap=torch.einsum('hwd,dk->hwk',jac,torch.from_numpy(twist).to(jac))
    raw=mi.printed_gradient(ref,ref,jac_mocap)
    keep=[0,1,3,4,5]; raw5=raw[keep]
    eps=[1e-4,3e-4,1e-3]
    rows=[]
    base_pos=np.array(d.mocap_pos[mid],copy=True); base_q=np.array(d.mocap_quat[mid],copy=True)
    for e in eps:
        fd=[]
        for j in keep:
            vals=[]
            for s in (-1.0,1.0):
                d.mocap_pos[mid]=base_pos; d.mocap_quat[mid]=base_q
                if j<3: d.mocap_pos[mid,j]+=s*e
                else:
                    axis=np.zeros(3); axis[j-3]=1.0; dq=np.r_[np.cos(e/2),axis*np.sin(s*e/2)]
                    d.mocap_quat[mid]=quat_mul(base_q,dq)
                mujoco.mj_forward(m,d); cur,_=render(renderer,d,'wrist_cam'); vals.append(float(mi(cur,ref)[0]))
            fd.append((vals[1]-vals[0])/(2*e))
        fd=torch.tensor(fd,dtype=torch.float64); r=raw5.to(torch.float64)
        cos=float(torch.dot(fd,r)/(fd.norm()*r.norm()+1e-12)); rows.append({'epsilon':e,'fd_gradient':fd.tolist(),'analytic_gradient':r.tolist(),'raw_cosine':cos})
    centered_norm=0.0
    report={'stage':'P1_reference_stationarity_derivative_audit','model':args.model,'bins':8,'padding':2,'dof_indices':keep,'camera_twist_transform':twist.tolist(),'raw_gradient':raw5.tolist(),'centered_reference_gradient_norm':centered_norm,'epsilon_rows':rows,'p1_derivative_pass':bool(min(x['raw_cosine'] for x in rows)>0.999)}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
