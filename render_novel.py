#!/usr/bin/env python3
"""Render an interpolated novel-view trajectory between training cameras.

Usage: python render_novel.py -m <model_dir> --iteration 30000 --outdir <dir> [--steps 4] [--rot slerp|fixed]
Encode the frames with e.g.: ffmpeg -framerate 24 -i <dir>/%04d.png -c:v libx264 -pix_fmt yuv420p out.mp4"""
import os, torch, numpy as np, torchvision
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation, Slerp
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_spherical, GaussianModel
from scene import Scene
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=30000, type=int)
parser.add_argument("--outdir", required=True, type=str)
parser.add_argument("--steps", default=10, type=int, help="interpolated frames per camera pair")
parser.add_argument("--rot", default="slerp", choices=["slerp", "fixed"],
                    help="orientation: slerp between cameras, or fixed to the first camera")
parser.add_argument("--ss", default=1, type=int, help="supersampling factor for anti-aliasing (2 = render 2x then downsample)")
args = get_combined_args(parser)

dataset, pipe = model.extract(args), pipeline.extract(args)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, panorama=True)
bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

cams = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
W, H = cams[0].image_width, cams[0].image_height
Wr, Hr = W * args.ss, H * args.ss  # render resolution (supersampled)
FOV = np.pi / 2
proj = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FOV, fovY=FOV).transpose(0, 1).cuda()

os.makedirs(args.outdir, exist_ok=True)
idx = 0
with torch.no_grad():
    for a, b in zip(cams[:-1], cams[1:]):
        Ca, Cb = a.camera_center.cpu().numpy(), b.camera_center.cpu().numpy()
        slerp = Slerp([0, 1], Rotation.from_matrix(np.stack([a.R, b.R])))
        for t in np.linspace(0, 1, args.steps, endpoint=False):
            R = cams[0].R if args.rot == "fixed" else slerp(t).as_matrix()
            C = (1 - t) * Ca + t * Cb
            T = -R.T @ C  # world-to-camera translation (R is cam-to-world)
            wv = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
            cam = MiniCam(Wr, Hr, FOV, FOV, 0.01, 100.0, wv, wv @ proj)
            img = render_spherical(cam, gaussians, pipe, bg)["render"]
            if args.ss > 1:  # supersample downsample = anti-aliasing
                img = torch.nn.functional.interpolate(img.unsqueeze(0), size=(H, W), mode="area").squeeze(0)
            torchvision.utils.save_image(img, os.path.join(args.outdir, f"{idx:04d}.png"))
            idx += 1
print(f"rendered {idx} frames -> {args.outdir}")
