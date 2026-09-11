"""E1-A0: free-camera smooth-plane MI scalar derivative audit.

This is deliberately a renderer / histogram audit, before the printed TRO
interaction matrix or wrist-camera actuation are introduced.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from mi_reward.control.tro_image_mi import TROImageMI

def render(xi: torch.Tensor, size: int) -> torch.Tensor:
    """Differentiable pinhole view of a fixed textured plane z=1 in world."""
    dtype=xi.dtype; device=xi.device
    grid=torch.linspace(-.42,.42,size,dtype=dtype,device=device); y,x=torch.meshgrid(grid,grid,indexing='ij')
    rx,ry,rz=xi[3:]; cx,sx=torch.cos(rx),torch.sin(rx); cy,sy=torch.cos(ry),torch.sin(ry); cz,sz=torch.cos(rz),torch.sin(rz)
    rxm=torch.stack((torch.stack((torch.ones_like(cx),torch.zeros_like(cx),torch.zeros_like(cx))),torch.stack((torch.zeros_like(cx),cx,-sx)),torch.stack((torch.zeros_like(cx),sx,cx))))
    rym=torch.stack((torch.stack((cy,torch.zeros_like(cy),sy)),torch.stack((torch.zeros_like(cy),torch.ones_like(cy),torch.zeros_like(cy))),torch.stack((-sy,torch.zeros_like(cy),cy))))
    rzm=torch.stack((torch.stack((cz,-sz,torch.zeros_like(cz))),torch.stack((sz,cz,torch.zeros_like(cz))),torch.stack((torch.zeros_like(cz),torch.zeros_like(cz),torch.ones_like(cz)))))
    rotation=rzm@rym@rxm; rays=rotation@torch.stack((x.reshape(-1),y.reshape(-1),torch.ones_like(x).reshape(-1)))
    scale=(1.-xi[2])/rays[2]; world=xi[:2,None]+rays[:2]*scale
    u,v=world[0],world[1]
    texture=127.5+45*torch.sin(19*u+7*v)+35*torch.cos(13*v-3*u)+25*torch.sin(31*u*v)
    return texture.reshape(size,size).clamp(0,255)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--size',type=int,default=64);p.add_argument('--epsilon-translation',type=float,default=1e-4);p.add_argument('--epsilon-rotation',type=float,default=1e-4);args=p.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    torch.set_default_dtype(torch.float64); estimator=TROImageMI(); reference=render(torch.zeros(6),args.size).detach()
    def scalar(x): return estimator(render(x,args.size),reference)[0]
    origin=torch.zeros(6,requires_grad=True); value=scalar(origin); analytic=torch.autograd.grad(value,origin)[0]
    image_pose_jacobian=torch.autograd.functional.jacobian(lambda pose: render(pose,args.size),torch.zeros(6))
    image_pose_hessian=torch.func.jacfwd(torch.func.jacfwd(lambda pose: render(pose,args.size)))(torch.zeros(6))
    printed=estimator.printed_gradient(render(torch.zeros(6),args.size),reference,image_pose_jacobian)
    printed_hessian=estimator.printed_hessian(render(torch.zeros(6),args.size),reference,image_pose_jacobian,image_pose_hessian)
    exact_probability_hessian=estimator.printed_hessian(render(torch.zeros(6),args.size),reference,image_pose_jacobian,image_pose_hessian,exact_derivative=True)
    hessian=torch.autograd.functional.hessian(scalar,torch.zeros(6))
    fd=[]; hfd=torch.empty((6,6),dtype=torch.float64); epsilons=[args.epsilon_translation]*3+[args.epsilon_rotation]*3
    for axis in range(6):
        eps=args.epsilon_translation if axis<3 else args.epsilon_rotation; plus=torch.zeros(6);minus=torch.zeros(6);plus[axis]=eps;minus[axis]=-eps
        fd.append((scalar(plus)-scalar(minus))/(2*eps))
    for left in range(6):
        for right in range(6):
            ei,ej=epsilons[left],epsilons[right]
            if left==right:
                plus=torch.zeros(6); minus=torch.zeros(6); plus[left]=ei; minus[left]=-ei
                hfd[left,right]=(scalar(plus)-2*value.detach()+scalar(minus))/(ei*ei)
            else:
                pp=torch.zeros(6); pm=torch.zeros(6); mp=torch.zeros(6); mm=torch.zeros(6)
                pp[left]=ei;pp[right]=ej;pm[left]=ei;pm[right]=-ej;mp[left]=-ei;mp[right]=ej;mm[left]=-ei;mm[right]=-ej
                hfd[left,right]=(scalar(pp)-scalar(pm)-scalar(mp)+scalar(mm))/(4*ei*ej)
    finite=torch.stack(fd); cos=float(torch.nn.functional.cosine_similarity(analytic[None],finite[None]).item()); rel=float(torch.linalg.norm(hessian-hfd)/torch.linalg.norm(hessian).clamp_min(1e-12)); _,audit=estimator(render(torch.zeros(6),args.size),reference)
    gradient_norm=float(torch.linalg.norm(analytic)); printed_cos=float(torch.nn.functional.cosine_similarity(printed[None],analytic[None]).item()); printed_rel=float(torch.linalg.norm(printed-analytic)/torch.linalg.norm(analytic).clamp_min(1e-12)); derivative_pass=cos>=.99 and rel<=.10 and printed_cos>=.99 and printed_rel<=.01
    printed_hessian_rel=float(torch.linalg.norm(printed_hessian-hessian)/torch.linalg.norm(hessian).clamp_min(1e-12))
    exact_probability_rel=float(torch.linalg.norm(exact_probability_hessian-hessian)/torch.linalg.norm(hessian).clamp_min(1e-12))
    report={'protocol':'e1_a0_free_camera_scalar_fd_v5','image_size':args.size,'mi_at_reference':float(value),'gradient_autograd':analytic.tolist(),'gradient_fd':finite.tolist(),'gradient_tro_printed':printed.tolist(),'gradient_autograd_fd_cosine':cos,'gradient_tro_printed_autograd_cosine':printed_cos,'gradient_tro_printed_relative_error':printed_rel,'gradient_norm_at_reference':gradient_norm,'hessian_numerical_relative_frobenius_error':rel,'hessian_tro_printed_relative_to_scalar':printed_hessian_rel,'hessian_exact_probability_relative_to_scalar':exact_probability_rel,'hessian_tro_printed_symmetry_error':float(torch.linalg.norm(printed_hessian-printed_hessian.T)),'hessian_scalar_symmetry_error':float(torch.linalg.norm(hessian-hessian.T)),'histogram_audit':audit.__dict__,'gates':{'probability_gradient_consistency_pass':derivative_pass,'printed_hessian_matches_scalar_pass':printed_hessian_rel<=.10,'exact_probability_hessian_matches_scalar_pass':exact_probability_rel<=.10,'reference_stationarity_pass':gradient_norm<=1e-6},'status':'AUDIT_COMPLETE'}
    args.output_dir.mkdir(parents=True);(args.output_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
