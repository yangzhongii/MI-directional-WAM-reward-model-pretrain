"""Evaluate structured contact-validity reward variants with trial split and bootstrap CI."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch

def auc(y,p):
 o=np.argsort(p);r=np.empty_like(o);r[o]=np.arange(len(o));pos=y==1;neg=~pos
 return float((r[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*neg.sum()+1e-9))
def ap(y,p):
 o=np.argsort(-p); yy=y[o]; return float(np.sum(np.cumsum(yy)[yy==1]/(np.where(yy==1)[0]+1)) / max(yy.sum(),1))
def main():
 apx=argparse.ArgumentParser();apx.add_argument('--data',required=True);apx.add_argument('--output',required=True);args=apx.parse_args();d=np.load(Path(args.data)/'frames.npz');m=json.load(open(Path(args.data)/'metadata.json'));y=np.array([int(x['label']=='valid_entry') for x in m['trials']]);
 # frame summaries keep groups explicit: rgb(4), proprio(7), ft(2), mi(1)
 rgb=d['rgb'].mean((2,3,4)); rgbstd=d['rgb'].std((2,3,4)); rgbf=np.stack([rgb,rgbstd],-1); prop=d['proprio'].mean(1); ft=d['ft'].mean(1); mi=d['mi'].mean(1,keepdims=True); groups={'rgb_only':rgbf.reshape(len(y),-1),'rgb_mi':np.concatenate([rgbf.reshape(len(y),-1),mi],1),'rgb_ft':np.concatenate([rgbf.reshape(len(y),-1),prop,ft],1),'full':np.concatenate([rgbf.reshape(len(y),-1),prop,ft,mi],1)}
 # Hold out one complete trajectory per condition (first cycle), preserving
 # all three contact-validity classes in the test set.
 te=np.arange(len(y))<3;tr=~te; out={}; rng=np.random.default_rng(5)
 for name,x in groups.items():
  z=torch.tensor(x,dtype=torch.float32);t=torch.tensor(y,dtype=torch.float32);mu=z[tr].mean(0);sd=z[tr].std(0).clamp_min(1e-5);z=(z-mu)/sd;w=torch.zeros(z.shape[1],requires_grad=True);b=torch.zeros((),requires_grad=True);opt=torch.optim.Adam([w,b],lr=.03)
  for _ in range(300):
   loss=torch.nn.functional.binary_cross_entropy_with_logits(z[tr]@w+b,t[tr]);opt.zero_grad();loss.backward();opt.step()
  p=torch.sigmoid(z[te]@w.detach()+b.detach()).numpy(); pred=p>.5; bal=.5*(np.mean(pred[y[te]==1]) + np.mean(~pred[y[te]==0])); ece=float(np.mean(np.abs(p-y[te]))); boots=[]
  for _ in range(1000):
   ii=rng.integers(0,len(p),len(p));
   if len(np.unique(y[te][ii]))>1: boots.append(auc(y[te][ii],p[ii]))
  out[name]={'auroc':auc(y[te],p),'auroc_ci95':[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],'auprc':ap(y[te],p),'balanced_accuracy':float(bal),'ece':ece,'test_trials':int(te.sum())}
 report={'protocol':'near_contact_trial_split_v2','positive_label':'valid_entry','class_counts':{k:int(sum(x['label']==k for x in m['trials'])) for k in ['valid_entry','jam','false_alignment']},'variants':out,'note':'Small custom-scene pilot; confidence intervals are wide and this is not Factory evidence.'};p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
