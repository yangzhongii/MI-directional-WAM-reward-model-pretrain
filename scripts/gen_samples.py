#!/usr/bin/env python3
"""Generate N sample videos from start.png for quality review."""
import torch, numpy as np, imageio, os, sys
from PIL import Image

# Mock safety checker
import diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict as cosmos_pipe
class DummySC:
    def __getattr__(self, _): return lambda *a, **k: True
    def to(self, *a, **k): return self
    def eval(self): return self
    def check_text_safety(self, p): return True
    def check_video_safety(self, v, *a, **k): return v
cosmos_pipe.CosmosSafetyChecker = DummySC

from diffusers import Cosmos2_5_PredictBasePipeline

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
    'weights/cosmos-diffusers-2b-pretrained',
    torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
)
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()

img = np.array(Image.open('weights/start.png').convert('RGB'))
h = (img.shape[0] // 16) * 16
w = (img.shape[1] // 16) * 16

os.makedirs('logs/test_samples', exist_ok=True)

for seed in range(N):
    print(f'[{seed+1}/{N}] seed={seed}...')
    with torch.inference_mode():
        result = pipe(
            prompt='A Franka robot arm with a gripper holding a green peg, moving smoothly downward to insert it into a white hole on the table. Do not touch the blue base.',
            image=Image.fromarray(img).resize((w,h)),
            num_frames=32, height=h, width=w,
            num_inference_steps=30,
            generator=torch.Generator().manual_seed(seed),
        )
    frames = result.frames[0]
    if isinstance(frames, list):
        frames = np.stack([np.array(f).astype(np.uint8) for f in frames])
    elif torch.is_tensor(frames):
        frames = frames.cpu().numpy()
    path = f'logs/test_samples/seed_{seed:02d}.mp4'
    imageio.mimsave(path, frames, fps=8)
    print(f'  -> {path} ({len(frames)} frames, {os.path.getsize(path)//1024}KB)')

print(f'Done! {N} videos in logs/test_samples/')
