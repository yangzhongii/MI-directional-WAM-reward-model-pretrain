"""Six-DOF point and image interaction matrices from Dame--Marchand (2011)."""
from __future__ import annotations
import torch

def point_interaction_matrix(height:int,width:int,fx:float,fy:float,cx:float,cy:float,depth:torch.Tensor)->torch.Tensor:
    """Return paper's Lx with shape [H,W,2,6]."""
    dtype=depth.dtype;device=depth.device
    v,u=torch.meshgrid(torch.arange(height,dtype=dtype,device=device),torch.arange(width,dtype=dtype,device=device),indexing='ij')
    x=(u-cx)/fx;y=(v-cy)/fy;z=depth
    zeros=torch.zeros_like(x);ones=torch.ones_like(x)
    row_x=torch.stack((-ones/z,zeros,x/z,x*y,-(ones+x*x),y),dim=-1)
    row_y=torch.stack((zeros,-ones/z,y/z,ones+y*y,-x*y,-x),dim=-1)
    return torch.stack((row_x,row_y),dim=-2)

def metric_image_gradient(image:torch.Tensor,fx:float,fy:float)->torch.Tensor:
    """Central image gradient converted from pixel to normalized metric coordinates."""
    gx=torch.empty_like(image);gy=torch.empty_like(image)
    gx[:,1:-1]=(image[:,2:]-image[:,:-2])/2;gx[:,0]=image[:,1]-image[:,0];gx[:,-1]=image[:,-1]-image[:,-2]
    gy[1:-1]=(image[2:]-image[:-2])/2;gy[0]=image[1]-image[0];gy[-1]=image[-1]-image[-2]
    return torch.stack((gx*fx,gy*fy),dim=-1)

def image_pose_jacobian(image:torch.Tensor,depth:torch.Tensor,fx:float,fy:float,cx:float,cy:float)->torch.Tensor:
    gradient=metric_image_gradient(image,fx,fy);interaction=point_interaction_matrix(*image.shape,fx,fy,cx,cy,depth)
    return torch.einsum('hwk,hwkd->hwd',gradient,interaction)
