#!/usr/bin/env python3
"""학습 카메라 12개 사이를 보간한 novel view 궤적을 equirect로 렌더 → PNG 시퀀스.
사용: python render_novel.py -m <model_dir> --iteration 30000 --outdir <dir>"""
import os, sys, torch, numpy as np, torchvision
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation, Slerp
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_spherical, GaussianModel
from scene import Scene
from scene.cameras import Camera
from utils.general_utils import safe_state

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=30000, type=int)
parser.add_argument("--outdir", required=True, type=str)
parser.add_argument("--steps", default=10, type=int)  # 카메라 쌍당 보간 프레임
args = get_combined_args(parser)
safe_state(True)

dataset, pipe = model.extract(args), pipeline.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, panorama=True)
bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

cams = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
H, W = cams[0].original_image.shape[1], cams[0].original_image.shape[2]
dummy = torch.zeros(3, H, W)

os.makedirs(args.outdir, exist_ok=True)
idx = 0
with torch.no_grad():
    for a, b in zip(cams[:-1], cams[1:]):
        Ca, Cb = -a.R @ a.T, -b.R @ b.T                      # 카메라 중심 (world)
        slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([a.R, b.R])))
        for t in np.linspace(0, 1, args.steps, endpoint=False):
            Rt = slerp(t).as_matrix()
            C = (1 - t) * Ca + t * Cb
            T = -Rt.T @ C                                     # w2c translation
            cam = Camera(colmap_id=0, R=Rt, T=T, FoVx=np.pi/2, FoVy=np.pi/2,
                         image=dummy, mask=None, gt_alpha_mask=None,
                         image_name=f"nv{idx:04d}", uid=idx, panorama=True)
            img = render_spherical(cam, gaussians, pipe, bg)["render"]
            torchvision.utils.save_image(img, os.path.join(args.outdir, f"{idx:04d}.png"))
            idx += 1
print(f"rendered {idx} frames -> {args.outdir}")
