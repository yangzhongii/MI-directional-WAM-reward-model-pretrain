"""Compare rendered image finite differences with the scene image Jacobian."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch, mujoco
from scipy.ndimage import gaussian_filter
from mi_reward.control.tro_image_interaction import image_pose_jacobian

def quat_mul(a,b):
    w,x,y,z=a; W,X,Y,Z=b
    return np.array([w*W-x*X-y*Y-z*Z,w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W])

def render(renderer,data):
    renderer.update_scene(data,camera='wrist_cam')
    return torch.from_numpy(gaussian_filter(renderer.render().astype(np.float64).mean(-1), sigma=1.0))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='assets/custom_task/scene/scene_peg_round_local_servo.xml'); ap.add_argument('--output',required=True); args=ap.parse_args()
    m=mujoco.MjModel.from_xml_path(args.model); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    renderer=mujoco.Renderer(m,64,64); full_ref=render(renderer,d); ref=full_ref[16:48,16:48]
    renderer.enable_depth_rendering(); renderer.update_scene(d,camera='wrist_cam'); depth=torch.from_numpy(renderer.render().astype(np.float64)); renderer.disable_depth_rendering()
    depth=depth[16:48,16:48]; valid=depth<10.0; safe_depth=depth.masked_fill(~valid, float(depth[valid].median()))
    cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'wrist_cam'); f=.5*64/np.tan(np.deg2rad(float(m.cam_fovy[cid]))/2)
    jac=image_pose_jacobian(ref,safe_depth,f,f,15.5,15.5)
    mid=int(m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'ee_target')]); basep=d.mocap_pos[mid].copy(); baseq=d.mocap_quat[mid].copy(); rows=[]
    for j in [0,1,2,3,4,5]:
        h=1e-2; ims=[]
        for s in [-1,1]:
            d.mocap_pos[mid]=basep; d.mocap_quat[mid]=baseq
            if j<3: d.mocap_pos[mid,j]+=s*h
            else:
                axis=np.zeros(3); axis[j-3]=1; d.mocap_quat[mid]=quat_mul(baseq,np.r_[np.cos(h/2),axis*np.sin(s*h/2)])
            mujoco.mj_forward(m,d); ims.append(render(renderer,d))
        fd=(ims[1][16:48,16:48]-ims[0][16:48,16:48])/(2*h); pred=jac[...,j]
        fd=fd[valid]; pred=pred[valid]
        rows.append({'dof':j,'cosine':float(torch.dot(fd.flatten(),pred.flatten())/(fd.norm()*pred.norm()+1e-12)),'fd_norm':float(fd.norm()),'pred_norm':float(pred.norm())})
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'stage':'scene_image_jacobian_audit','rows':rows},indent=2)+'\n'); print(json.dumps({'rows':rows},indent=2))
if __name__=='__main__': main()
