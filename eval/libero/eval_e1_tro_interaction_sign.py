"""E1-A1: audit TRO Lx magnitude and pose-convention sign on the free-camera plane."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from eval_e1_tro_synthetic_pose_fd import render
from mi_reward.control.tro_image_interaction import image_pose_jacobian
from mi_reward.control.tro_image_mi import TROImageMI

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--size',type=int,default=240);args=p.parse_args()
 if args.output_dir.exists():raise FileExistsError(args.output_dir)
 torch.set_default_dtype(torch.float64);zero=torch.zeros(6);image=render(zero,args.size);exact=torch.func.jacfwd(lambda x:render(x,args.size))(zero)
 focal=(args.size-1)/.84;tro=image_pose_jacobian(image,torch.ones_like(image),focal,focal,(args.size-1)/2,(args.size-1)/2)
 def stats(candidate):
  e=exact.reshape(-1,6);c=candidate.reshape(-1,6);per=[]
  for axis in range(6):per.append(float(torch.nn.functional.cosine_similarity(e[:,axis][None],c[:,axis][None]).item()))
  return {'global_cosine':float(torch.nn.functional.cosine_similarity(e.reshape(1,-1),c.reshape(1,-1)).item()),'per_axis_cosine':per,'relative_error':float(torch.linalg.norm(e-c)/torch.linalg.norm(e))}
 paper=stats(tro);twc=stats(-tro);chosen='paper_camera_velocity' if paper['global_cosine']>twc['global_cosine'] else 'negative_paper_Lx_for_Twc_body_increment'
 estimator=TROImageMI(); reference=image.detach(); pose=torch.zeros(6,requires_grad=True); scalar=estimator(render(pose,args.size),reference)[0]; scalar_gradient=torch.autograd.grad(scalar,pose)[0]; printed_gradient=estimator.printed_gradient(image,reference,-tro); mi_cos=float(torch.nn.functional.cosine_similarity(scalar_gradient[None],printed_gradient[None]).item());mi_rel=float(torch.linalg.norm(scalar_gradient-printed_gradient)/torch.linalg.norm(scalar_gradient))
 report={'protocol':'e1_a1_tro_interaction_sign_v2','image_size':args.size,'intrinsics':{'fx':focal,'fy':focal,'cx':(args.size-1)/2,'cy':(args.size-1)/2},'paper_Lx':paper,'negative_paper_Lx':twc,'pose_convention_lock':chosen,'mi_gradient_chain':{'scalar_autograd':scalar_gradient.tolist(),'tro_printed_with_negative_Lx':printed_gradient.tolist(),'cosine':mi_cos,'relative_error':mi_rel},'status':'PASS' if max(paper['global_cosine'],twc['global_cosine'])>=.99 and mi_cos>=.99 and mi_rel<=.01 else 'NO_GO'}
 args.output_dir.mkdir(parents=True);(args.output_dir/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
