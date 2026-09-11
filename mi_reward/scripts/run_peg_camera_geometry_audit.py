"""Check wrist-camera visibility, depth coverage, and intrinsic consistency."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import mujoco

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='assets/custom_task/scene/scene_peg_round_local_servo.xml'); ap.add_argument('--output',required=True); args=ap.parse_args()
    m=mujoco.MjModel.from_xml_path(args.model); d=mujoco.MjData(m); mujoco.mj_forward(m,d)
    cid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,'wrist_cam'); hid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'insertion_hole')
    r=mujoco.Renderer(m,128,128); r.update_scene(d,camera='wrist_cam'); rgb=r.render(); r.enable_depth_rendering(); r.update_scene(d,camera='wrist_cam'); depth=r.render(); r.disable_depth_rendering()
    fovy=float(m.cam_fovy[cid]); fy=0.5*128/np.tan(np.deg2rad(fovy)/2); fx=fy
    pos=d.cam_xpos[cid]; mat=d.cam_xmat[cid].reshape(3,3); hole=d.xpos[hid]; rel=mat@(hole-pos)
    report={'camera_id':int(cid),'camera_position':pos.tolist(),'camera_matrix_rows':mat.tolist(),'hole_world_position':hole.tolist(),'hole_camera_coordinates':rel.tolist(),'fovy_deg':fovy,'fx':float(fx),'fy':float(fy),'rgb_std':float(rgb.std()),'depth_min':float(depth.min()),'depth_max':float(depth.max()),'depth_unique':int(np.unique(depth).size),'valid_depth_fraction':float(np.mean(depth < 0.99*depth.max()))}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
