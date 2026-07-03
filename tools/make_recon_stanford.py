#!/usr/bin/env python3
"""Stanford 2D-3D-S pano poses + global_xyz -> OpenSfM reconstruction.json for 360-gaussian-splatting.
No SfM: uses ground-truth camera_rt_matrix (verified world->cam, X_cam=R*X_world+t) and
per-pixel global_xyz (.exr) as the init point cloud. Feeds train.py -s <OUT> --panorama.

Usage: python tools/make_recon_stanford.py <pano_dir> <out_dir>
<pano_dir> is an area's pano/ folder containing rgb/, pose/, global_xyz/."""
import os, sys, json, glob
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from scipy.spatial.transform import Rotation

if len(sys.argv) != 3:
    sys.exit(__doc__)
SRC, OUT = sys.argv[1], sys.argv[2]
OUT_RES = (2048, 1024)   # ponytail: train image (W,H); drop to (1024,512) if 12GB OOM
PTS_PER_PANO = 20000
np.random.seed(0)

os.makedirs(OUT + "/images", exist_ok=True)
pose_files = sorted(glob.glob(SRC + "/pose/*.json"))
assert pose_files, "no pose json found"

shots, all_pts, all_cols, cam_centers = {}, [], [], []
for pf in pose_files:
    pose = json.load(open(pf))
    uuid = pose["camera_uuid"]
    rgb_path = glob.glob(f"{SRC}/rgb/camera_{uuid}_*rgb.png")[0]
    xyz_path = glob.glob(f"{SRC}/global_xyz/camera_{uuid}_*global_xyz.exr")[0]
    name = os.path.basename(rgb_path)

    rt = np.array(pose["camera_rt_matrix"], dtype=np.float64)   # 3x4 [R|t], world->cam
    R, t = rt[:, :3], rt[:, 3]
    rotvec = Rotation.from_matrix(R).as_rotvec()                # angle-axis of R (w2c), OpenSfM convention
    shots[name] = {"camera": "spherical", "rotation": rotvec.tolist(), "translation": t.tolist()}
    cam_centers.append(-R.T @ t)                                # camera center in world (== camera_location)

    xyz = cv2.imread(xyz_path, cv2.IMREAD_UNCHANGED)[..., ::-1]   # HxWx3 float32, BGR -> X,Y,Z
    bgr = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)             # BGRA/BGR
    if bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(f"{OUT}/images/{name}", cv2.resize(bgr, OUT_RES, interpolation=cv2.INTER_AREA))  # train image

    P = xyz.reshape(-1, 3)                                       # float32
    valid = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 1e-6)    # drop NaN/inf and (0,0,0) no-return
    idx = np.where(valid)[0]
    if len(idx) > PTS_PER_PANO:
        idx = np.random.choice(idx, PTS_PER_PANO, replace=False)
    all_pts.append(P[idx].astype(np.float64))
    all_cols.append(bgr.reshape(-1, 3)[idx][:, ::-1])            # BGR->RGB, selected rows only

pts = np.concatenate(all_pts)
cols = np.concatenate(all_cols)
cam_centers = np.array(cam_centers)

# --- sanity (non-circular): camera centers must sit inside the point-cloud bbox ---
lo, hi = pts.min(0), pts.max(0)
inside = ((cam_centers >= lo) & (cam_centers <= hi)).all(1)
print(f"points={len(pts)}  bbox_lo={lo.round(2)}  bbox_hi={hi.round(2)}  extent={(hi-lo).round(2)}")
print(f"cam_centers inside bbox: {inside.sum()}/{len(cam_centers)}  (expect all; else XYZ channel/frame wrong)")
print(f"cam0 center={cam_centers[0].round(3)}")

coords, colors = pts.tolist(), cols.tolist()
points = {str(i): {"coordinates": coords[i], "color": colors[i]} for i in range(len(coords))}
recon = [{
    "reference_lla": {"latitude": 0.0, "longitude": 0.0, "altitude": 0.0},
    "cameras": {"spherical": {"projection_type": "spherical", "width": 4096, "height": 2048}},
    "shots": shots,
    "points": points,
}]
json.dump(recon, open(f"{OUT}/reconstruction.json", "w"))
print(f"wrote {OUT}/reconstruction.json  ({len(shots)} shots)  and {len(shots)} images @ {OUT_RES}")
