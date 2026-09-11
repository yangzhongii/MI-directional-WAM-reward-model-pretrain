"""Single-task corrected v8 M1 experiment with preregistered metrics."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from mi_reward.closed_loop.libero_env import apply_appearance_shift
from mi_reward.data.libero_privileged import extract_privileged_records, resolve_libero_task
from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor
from eval.libero.run_v8_m1_smoke import M1Head, stage_and_factor

VARIANTS=("none","dark_warm","bright_cool","low_contrast")
GROUPS=("rank_stage","physical_regression","conditional_mi","physical_regression_conditional_mi")
CONTINUOUS=(0,1,2,3,4,5,6,7,8,11); BINARY=(9,10)

def load_data(root,demo,extractor,mean=None,std=None):
    path,bddl,language=resolve_libero_task(root,"libero_spatial",0)
    with h5py.File(path,"r") as f:
        g=f["data"][demo]; images=g["obs"]["agentview_rgb"][:]
        priv=extract_privileged_records(bddl_path=bddl,states=g["states"][:],actions=g["actions"][:],rewards=g["rewards"][:],dones=g["dones"][:])
    stage,factor,progress=zip(*(stage_and_factor(x) for x in priv["frames"]))
    physical=torch.tensor(np.asarray(factor),dtype=torch.float32)
    if mean is not None: physical=(physical-mean)/std
    features={}
    for v in VARIANTS:
        ims=[apply_appearance_shift(im,v) for im in images]
        tok=extractor.extract_trajectory_tokens_images(ims,language)
        features[v]=tok.float().mean(1)
    return {"features":features,"goal":features["none"][-1],"stage":torch.tensor(stage),"physical":physical,"progress":torch.tensor(progress,dtype=torch.float32)}

def cmi(model,h,physical,stage,state,tau=.2):
    q=F.normalize(model.visual_proj(h),dim=-1); k=F.normalize(model.physical_proj(physical),dim=-1); logits=q@k.T/tau
    valid=stage[:,None].eq(stage[None,:]); pos=valid & state[:,None].eq(state[None,:]); logits=logits.masked_fill(~valid,-1e4)
    a=(torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~pos,-1e4),1)).mean()
    b=(torch.logsumexp(logits.T,1)-torch.logsumexp(logits.T.masked_fill(~pos.T,-1e4),1)).mean()
    return (a+b)/2

def batches(data,device,nstates=24):
    n=len(data["stage"])
    # Feature tensors are held on CPU; sample/index on CPU, then transfer the
    # assembled same-state four-variant minibatch to the model device.
    ids=torch.randint(n,(min(nstates,n),))
    xs=[]; st=[]; ph=[]; pr=[]; sid=[]
    for v in VARIANTS:
        xs.append(data["features"][v][ids]); st.append(data["stage"][ids]); ph.append(data["physical"][ids]); pr.append(data["progress"][ids]); sid.append(ids)
    return (torch.cat(xs).to(device), torch.cat(st).to(device),
            torch.cat(ph).to(device), torch.cat(pr).to(device),
            torch.cat(sid).to(device))

def pair_metrics(value,progress,stage):
    n=len(value); i,j=torch.triu_indices(n,n,1,device=value.device); dp=progress[i]-progress[j]; dv=value[i]-value[j]; valid=dp.abs()>1e-6
    def acc(mask): return float(((dv*dp>0)[mask]).float().mean()) if bool(mask.any()) else float("nan")
    all_acc=acc(valid); same=valid & stage[i].eq(stage[j]); within=acc(same)
    absdp=dp.abs(); hard=same & (absdp<=torch.quantile(absdp[same],.25)) if bool(same.any()) else same
    return {"all_pair_accuracy":all_acc,"within_stage_pair_accuracy":within,"hard_pair_accuracy":acc(hard),"pair_count_all":int(valid.sum()),"pair_count_within_stage":int(same.sum()),"pair_count_hard":int(hard.sum())}

def flip_rates(raw,variants):
    n=len(raw); i,j=torch.triu_indices(n,n,1); base=raw[i]-raw[j]; valid=base.abs()>1e-7; out={}
    for v,x in variants.items():
        if v=="none": continue
        shifted=x[i]-x[j]; out[v]=float(((base[valid]*shifted[valid])<0).float().mean()) if bool(valid.any()) else float("nan")
    return out

def ridge_r2(pred,y):
    x=torch.cat((pred,torch.ones(len(pred),1)),1); w=torch.linalg.lstsq(x,y).solution; yh=torch.cat((pred,torch.ones(len(pred),1)),1)@w
    return (1-(yh-y).square().sum(0)/(y-y.mean(0)).square().sum(0).clamp_min(1e-8)).tolist()

def auroc(pred,y):
    order=torch.argsort(pred); ranks=torch.empty_like(order,dtype=torch.float32); ranks[order]=torch.arange(len(pred),dtype=torch.float32); pos=y>0.5; neg=~pos
    if not bool(pos.any()) or not bool(neg.any()): return float("nan")
    return float(((ranks[pos].sum()-pos.sum()*(pos.sum()-1)/2)/(pos.sum()*neg.sum())).item())

def evaluate(model,data,device):
    model.eval(); values={}; stages={}; physical={}; hidden={}
    with torch.no_grad():
        for v in VARIANTS:
            val,sl,ph,h=model(data["features"][v].to(device),data["goal"].to(device).expand(len(data["stage"]),-1)); values[v]=val.cpu(); stages[v]=sl.argmax(-1).cpu(); physical[v]=ph.cpu(); hidden[v]=h.cpu()
    metrics=pair_metrics(values["none"],data["progress"],data["stage"]); metrics["pair_flip_rate"]=flip_rates(values["none"],values); metrics["stage_accuracy"]=float((stages["none"]==data["stage"]).float().mean()); metrics["physical_r2_by_factor"]=ridge_r2(hidden["none"],data["physical"])
    metrics["physical_r2_continuous_mean"]=float(np.mean([metrics["physical_r2_by_factor"][i] for i in CONTINUOUS])); metrics["physical_auroc_by_factor"]={str(i):auroc(physical["none"][:,i],data["physical"][:,i]) for i in BINARY}
    return metrics,{"value":values,"stage_prediction":stages,"physical_prediction":physical,"hidden":hidden}

def train_group(group,seed,train,val,device,lam,steps=700):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); dim=train["features"]["none"].shape[-1]; model=M1Head(dim,train["physical"].shape[-1],group in ("conditional_mi","physical_regression_conditional_mi")).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    td={k:(v.to(device) if isinstance(v,torch.Tensor) else {q:x.to(device) for q,x in v.items()}) for k,v in train.items()}; best=None; best_score=-1
    for _ in range(steps):
        x,st,ph,pr,sid=batches(train,device); value,sl,phat,h=model(x,td["goal"].expand(len(x),-1)); other=torch.roll(torch.arange(len(x),device=device),1); rank=F.softplus(-torch.sign(pr-pr[other])*(value-value[other])).mean(); loss=rank+0.5*F.cross_entropy(sl,st)
        if group in ("physical_regression","physical_regression_conditional_mi"): loss=loss+0.2*F.mse_loss(phat,ph)
        if group in ("conditional_mi","physical_regression_conditional_mi"): loss=loss+lam*cmi(model,h,ph,st,sid)
        opt.zero_grad(); loss.backward(); opt.step()
    metric,_=evaluate(model,val,device); score=metric["all_pair_accuracy"]
    return model,metric

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--steps",type=int,default=700); a=p.parse_args();
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    root=ROOT; extractor=DINOv3FeatureExtractor(model_path=str(root/".venv/models/dinov3-vitb16-pretrain-lvd1689m"),device="cuda",image_size=224,strict=True)
    raw=load_data(root,"demo_0",extractor); mean=raw["physical"].mean(0); std=raw["physical"].std(0).clamp_min(1e-4); train=load_data(root,"demo_0",extractor,mean,std); val=load_data(root,"demo_1",extractor,mean,std); test=load_data(root,"demo_10",extractor,mean,std); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambdas=(0.03,0.1,0.3); records=[]; predictions={}
    for group in GROUPS:
        for seed in (7,17,27):
            candidates=[]
            for lam in (lambdas if "conditional_mi" in group else (0.0,)):
                model,vm=train_group(group,seed,train,val,device,lam,a.steps); candidates.append((vm["all_pair_accuracy"],lam,model,vm))
            _,chosen_lam,model,val_metric=max(candidates,key=lambda x:(x[0],-x[1])); test_metric,test_pred=evaluate(model,test,device); train_metric,_=evaluate(model,train,device)
            records.append({"group":group,"seed":seed,"chosen_lambda_mi":chosen_lam,"validation":val_metric,"test":test_metric,"train":train_metric}); predictions[f"{group}_seed_{seed}"]={"chosen_lambda_mi":chosen_lam,"test":test_pred}
    summary={g:{"test_all_pair_accuracy":float(np.mean([x["test"]["all_pair_accuracy"] for x in records if x["group"]==g])),"test_within_stage_pair_accuracy":float(np.mean([x["test"]["within_stage_pair_accuracy"] for x in records if x["group"]==g])),"test_hard_pair_accuracy":float(np.mean([x["test"]["hard_pair_accuracy"] for x in records if x["group"]==g])),"test_pair_flip_rate":{v:float(np.mean([x["test"]["pair_flip_rate"][v] for x in records if x["group"]==g])) for v in VARIANTS if v!="none"}} for g in GROUPS}
    a.output_dir.mkdir(parents=True); (a.output_dir/"predictions").mkdir(); torch.save(predictions,a.output_dir/"predictions"/"selected_models_frame_outputs.pt"); report={"protocol":"v8_m1_corrected_single_task_v1","split":{"train":"demo_0","validation":"demo_1_lambda_selection","heldout":"demo_10_single_test"},"groups":GROUPS,"variants":VARIANTS,"records":records,"summary":summary,"heldout_test_count":len(test["stage"]),"formal_gate":"single_task_corrected_diagnostic_pending_multi_task"}; (a.output_dir/"results.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__": main()
