"""v8 M1 matched-state appearance-shift gate on frozen DINOv3 features."""
from __future__ import annotations

import argparse, json, random
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mi_reward.closed_loop.libero_env import apply_appearance_shift
from mi_reward.data.libero_privileged import extract_privileged_records, resolve_libero_task
from mi_reward.features.dino_v3_extractor import DINOv3FeatureExtractor
from eval.libero.run_v8_m1_smoke import M1Head, pair_accuracy, probe_r2, stage_and_factor


def load_split(root: Path, demo: str, split: str, extractor: DINOv3FeatureExtractor, mean=None, std=None):
    demo_path, bddl, language = resolve_libero_task(root, "libero_spatial", 0)
    with h5py.File(demo_path, "r") as handle:
        group = handle["data"][demo]
        images = group["obs"]["agentview_rgb"][:]
        privileged = extract_privileged_records(bddl_path=bddl, states=group["states"][:], actions=group["actions"][:], rewards=group["rewards"][:], dones=group["dones"][:])
    stages, factors, progress = zip(*(stage_and_factor(frame) for frame in privileged["frames"]))
    physical = torch.tensor(np.asarray(factors), dtype=torch.float32)
    if mean is not None: physical = (physical-mean)/std
    variants = ("none", "dark_warm", "bright_cool", "low_contrast")
    features = {}
    for variant in variants:
        shifted = [apply_appearance_shift(image, variant) for image in images]
        tokens = extractor.extract_trajectory_tokens_images(shifted, language)
        features[variant] = tokens.float().mean(dim=1)
    goal = features["none"][-1]
    return {"features": features, "goal": goal, "stage": torch.tensor(stages), "physical": physical, "progress": torch.tensor(progress), "state_id": torch.arange(len(stages)), "split": split}


def cmi_multi(model, hidden, physical, stage, state_id, tau=0.2):
    q=F.normalize(model.visual_proj(hidden),dim=-1); k=F.normalize(model.physical_proj(physical),dim=-1)
    logits=q@k.T/tau; valid=(stage[:,None]==stage[None,:]); positive=(state_id[:,None]==state_id[None,:]) & valid
    logits=logits.masked_fill(~valid, -1e4); logden=torch.logsumexp(logits,dim=1); logpos=torch.logsumexp(logits.masked_fill(~positive,-1e4),dim=1)
    logits_t=logits.T; logden_t=torch.logsumexp(logits_t,dim=1); logpos_t=torch.logsumexp(logits_t.masked_fill(~positive.T,-1e4),dim=1)
    return 0.5*((logden-logpos).mean()+(logden_t-logpos_t).mean())


def run(seed, enabled, train, held, steps, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dim=train["features"]["none"].shape[-1]; model=M1Head(dim,train["physical"].shape[-1],enabled).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    def flatten(data):
        xs=[]; stages=[]; phys=[]; prog=[]; ids=[]
        for variant,x in data["features"].items(): xs.append(x); stages.append(data["stage"]); phys.append(data["physical"]); prog.append(data["progress"]); ids.append(data["state_id"])
        return torch.cat(xs),torch.cat(stages),torch.cat(phys),torch.cat(prog),torch.cat(ids)
    xv,st,ph,pr,sid=flatten(train); xv=xv.to(device); st=st.to(device); ph=ph.to(device); pr=pr.to(device); sid=sid.to(device); goal=train["goal"].to(device)
    for _ in range(steps):
        idx=torch.randint(len(xv),(min(128,len(xv)),),device=device); value,stage_logits,phat,hidden=model(xv[idx],goal.expand(len(idx),-1)); other=torch.roll(torch.arange(len(idx),device=device),1)
        rank=F.softplus(-torch.sign(pr[idx]-pr[idx][other])*(value-value[other])).mean(); sl=F.cross_entropy(stage_logits,st[idx]); pl=F.mse_loss(phat,ph[idx]); ml=cmi_multi(model,hidden,ph[idx],st[idx],sid[idx]) if enabled else value.new_zeros(())
        loss=rank+0.5*sl+0.2*pl+0.1*ml; opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        held_values={v:model(held["features"][v].to(device),held["goal"].to(device).expand(len(held["stage"]),-1))[0].cpu() for v in held["features"]}
        base=held_values["none"]; pair=pair_accuracy(base,held["progress"]); flips={v:float((((base-base.flip(0))*(held_values[v]-held_values[v].flip(0))<0).float().mean())) for v in held_values if v!="none"}
        _,_,_,trh=model(train["features"]["none"].to(device),goal.expand(len(train["stage"]),-1)); _,_,_,teh=model(held["features"]["none"].to(device),held["goal"].to(device).expand(len(held["stage"]),-1))
    return {"held_pair_accuracy":pair,"pair_flip_rate":flips,"held_physical_probe_r2":probe_r2(trh,train["physical"].to(device),teh,held["physical"].to(device))}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--steps",type=int,default=1000); a=p.parse_args();
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    root=Path(__file__).resolve().parents[2]; extractor=DINOv3FeatureExtractor(model_path=str(root/".venv/models/dinov3-vitb16-pretrain-lvd1689m"),device="cuda",image_size=224,strict=True)
    raw=[load_split(root,n,"train",extractor) for n in ("demo_0","demo_1")]; mean=torch.cat([x["physical"] for x in raw]).mean(0); std=torch.cat([x["physical"] for x in raw]).std(0).clamp_min(1e-4)
    train_parts=[load_split(root,n,"train",extractor,mean,std) for n in ("demo_0","demo_1")]; held=load_split(root,"demo_10","heldout",extractor,mean,std)
    train={"features":{v:torch.cat([x["features"][v] for x in train_parts]) for v in train_parts[0]["features"]},"goal":train_parts[0]["goal"],"stage":torch.cat([x["stage"] for x in train_parts]),"physical":torch.cat([x["physical"] for x in train_parts]),"progress":torch.cat([x["progress"] for x in train_parts]),"state_id":torch.cat([x["state_id"]+i*1000 for i,x in enumerate(train_parts)])}
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); records=[]
    for seed in (7,17,27):
        for method,enabled in (("B5_base",False),("M1_physical_cmi",True)): records.append({"seed":seed,"method":method,**run(seed,enabled,train,held,a.steps,device)})
    methods={m:[x for x in records if x["method"]==m] for m in ("B5_base","M1_physical_cmi")}; summary={m:{"held_pair_accuracy":float(np.mean([x["held_pair_accuracy"] for x in xs])),"held_physical_probe_r2":float(np.mean([x["held_physical_probe_r2"] for x in xs])),"pair_flip_rate":{v:float(np.mean([x["pair_flip_rate"][v] for x in xs])) for v in xs[0]["pair_flip_rate"]}} for m,xs in methods.items()}
    report={"protocol":"v8_m1_matched_state_appearance_v1","task":"libero_spatial/task-00","variants":["none","dark_warm","bright_cool","low_contrast"],"feature_encoder":"local_dinov3_vitb16_frozen","records":records,"summary":summary,"formal_gate":"pending_success_failure_and_multi_task_extension"}; a.output_dir.mkdir(parents=True); (a.output_dir/"results.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__": main()
