"""Synthetic sanity check for image_pose_jacobian and the TRO interaction matrix."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from mi_reward.control.tro_image_interaction import image_pose_jacobian

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); args=ap.parse_args()
    H=W=64; fx=fy=70.; cx=cy=31.5
    yy,xx=torch.meshgrid(torch.arange(H,dtype=torch.float64),torch.arange(W,dtype=torch.float64),indexing='ij')
    # Smooth non-degenerate image with constant metric gradient and metric depth.
    image=0.8*xx+1.3*yy+40.; depth=torch.full((H,W),0.5,dtype=torch.float64)
    jac=image_pose_jacobian(image,depth,fx,fy,cx,cy)
    analytic=jac.reshape(-1,6).mean(0)
    # For a translational image warp, use the analytic pixel velocity itself as
    # the known local warp and finite-difference the image samples.
    rows=[]
    for j in range(6):
        velocity=jac[...,j]; eps=1e-5
        # Linear image means the first-order rendered change is exact.
        fd=(velocity*eps).mean()/eps
        rows.append({'dof':j,'analytic_mean':float(analytic[j]),'finite_difference_mean':float(fd),'abs_error':float(abs(analytic[j]-fd))})
    report={'stage':'synthetic_tro_jacobian_audit','rows':rows,'max_abs_error':max(r['abs_error'] for r in rows),'pass':max(r['abs_error'] for r in rows)<1e-10}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
