#!/usr/bin/env python3
"""OpenSfM dense(merged.ply) 점군으로 reconstruction.json의 sparse points 교체.
이미지는 원본 데이터 dir을 symlink. 사용: python make_dense_recon.py [SRC_dataset_dir] [OUT_dir]"""
import os, sys, json
import numpy as np
from plyfile import PlyData

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/ingon/1/ind-bermuda-opensfm/data/kungsgatanparken"
PLY = f"{SRC}/undistorted/depthmaps/merged.ply"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/ingon/1/data/kungs_dense"
MAX_PTS = 400_000
np.random.seed(0)

os.makedirs(OUT, exist_ok=True)
if not os.path.islink(f"{OUT}/images"):
    os.symlink(f"{SRC}/images", f"{OUT}/images")

ply = PlyData.read(PLY)
v = ply["vertex"].data
pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
cols = np.stack([v["red"], v["green"], v["blue"]], 1).astype(int)
valid = np.isfinite(pts).all(1)
idx = np.where(valid)[0]
print(f"dense points: {len(pts)} (valid {len(idx)})")
if len(idx) > MAX_PTS:
    idx = np.random.choice(idx, MAX_PTS, replace=False)

recon = json.load(open(f"{SRC}/reconstruction.json"))
rec = recon[0] if isinstance(recon, list) else recon
rec["points"] = {str(i): {"coordinates": pts[j].tolist(),
                          "color": cols[j].tolist()} for i, j in enumerate(idx)}
json.dump(recon if isinstance(recon, list) else [rec], open(f"{OUT}/reconstruction.json", "w"))
print(f"wrote {OUT}/reconstruction.json  shots={len(rec['shots'])}  points={len(idx)}")
