#!/usr/bin/env python3
"""Replace the sparse SfM points in reconstruction.json with an OpenSfM MVS dense cloud.

Usage: python tools/make_dense_recon.py <sfm_dataset_dir> <out_dir> [max_pts]
<sfm_dataset_dir> must contain reconstruction.json, images/, undistorted/depthmaps/merged.ply
(produced by `bin/opensfm undistort` + `bin/opensfm compute_depthmaps`).
max_pts: number of MVS points to keep (default 400000; use 1500000 for dense/high quality)."""
import os, sys, json
import numpy as np
from plyfile import PlyData

if len(sys.argv) not in (3, 4):
    sys.exit(__doc__)
SRC, OUT = sys.argv[1], sys.argv[2]
MAX_PTS = int(sys.argv[3]) if len(sys.argv) == 4 else 400_000
np.random.seed(0)

os.makedirs(OUT, exist_ok=True)
if not os.path.islink(f"{OUT}/images"):
    os.symlink(os.path.abspath(f"{SRC}/images"), f"{OUT}/images")

v = PlyData.read(f"{SRC}/undistorted/depthmaps/merged.ply")["vertex"].data
pts = np.stack([v["x"], v["y"], v["z"]], 1)
cols = np.stack([v["red"], v["green"], v["blue"]], 1)
idx = np.where(np.isfinite(pts).all(1))[0]
print(f"dense points: {len(pts)} (valid {len(idx)})")
if len(idx) > MAX_PTS:
    idx = np.random.choice(idx, MAX_PTS, replace=False)
coords, colors = pts[idx].tolist(), cols[idx].tolist()

recon = json.load(open(f"{SRC}/reconstruction.json"))
if not isinstance(recon, list):
    recon = [recon]
recon[0]["points"] = {str(i): {"coordinates": c, "color": rgb}
                      for i, (c, rgb) in enumerate(zip(coords, colors))}
json.dump(recon, open(f"{OUT}/reconstruction.json", "w"))
print(f"wrote {OUT}/reconstruction.json  shots={len(recon[0]['shots'])}  points={len(idx)}")
