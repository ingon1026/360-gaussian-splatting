import torch, os, math, numpy as np
from argparse import ArgumentParser
from scene import Scene
from gaussian_renderer import render_spherical, GaussianModel
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
import torchvision

parser = ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
parser.add_argument("--iteration", default=30000, type=int)
parser.add_argument("--frames", default=150, type=int)
parser.add_argument("--outdir", required=True)
args = get_combined_args(parser)
safe_state(True)
dataset = model.extract(args); pipe = pipeline.extract(args)
os.makedirs(args.outdir, exist_ok=True)

gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, panorama=True)
cams = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
W, H = cams[0].image_width, cams[0].image_height
FOV = math.pi/2
print(f"{len(cams)} train cams, render {W}x{H}, {args.frames} frames")

# camera centers (world) + keep first cam's orientation constant (stable equirect north)
centers = np.array([c.camera_center.detach().cpu().numpy() for c in cams])
R0 = np.array(cams[0].R)  # c2w rotation of cam0

# resample the 12-point path to N points by arc length (smooth dolly)
seg = np.linalg.norm(np.diff(centers, axis=0), axis=1)
cum = np.concatenate([[0], np.cumsum(seg)])
tt = np.linspace(0, cum[-1], args.frames)
path = np.stack([np.interp(tt, cum, centers[:, k]) for k in range(3)], axis=1)

Rwc0 = R0.T
bg = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
with torch.no_grad():
    for i, C in enumerate(path):
        T = (-Rwc0 @ C).astype(np.float32)
        wv = torch.tensor(getWorld2View2(R0, T)).transpose(0,1).cuda()
        proj = getProjectionMatrix(znear=0.01, zfar=100, fovX=FOV, fovY=FOV).transpose(0,1).cuda()
        full = (wv.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
        cam = MiniCam(W, H, FOV, FOV, 0.01, 100, wv, full)
        img = torch.clamp(render_spherical(cam, gaussians, pipe, bg)["render"], 0, 1)
        torchvision.utils.save_image(img, os.path.join(args.outdir, f"{i:04d}.png"))
        if i % 30 == 0: print(f"  frame {i}/{args.frames}")
print("done:", args.outdir)
