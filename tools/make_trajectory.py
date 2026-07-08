#!/usr/bin/env python3
# reconstruction.json → trajectory png (점군 위 카메라 경로, top-down view)
# 사용: python make_trajectory.py <reconstruction.json> <out.png> [제목]
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rec = json.load(open(sys.argv[1]))[0]
out = sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "trajectory"

pts = np.array([p["coordinates"] for p in rec["points"].values()], float)
if len(pts) > 120000:
    pts = pts[np.random.RandomState(0).choice(len(pts), 120000, replace=False)]

cams = []
for s in rec["shots"].values():
    r = np.array(s["rotation"], float); t = np.array(s["translation"], float)
    th = np.linalg.norm(r)
    if th < 1e-8:
        R = np.eye(3)
    else:
        k = r / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    cams.append(-R.T @ t)
cams = np.array(cams)

# top-down 평면 자동 선택: 카메라 분산 최소축 = 높이(up), 나머지 두 축이 바닥평면
a, b = [i for i in range(3) if i != int(np.argmin(cams.std(0)))]
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(pts[:, a], pts[:, b], s=0.2, c="#999", alpha=0.25, linewidths=0)
ax.plot(cams[:, a], cams[:, b], "-", c="#e34a3c", lw=1.2, alpha=0.8)
ax.scatter(cams[:, a], cams[:, b], s=14, c="#e34a3c", zorder=3, edgecolors="white", linewidths=0.4)
ax.scatter(cams[0, a], cams[0, b], s=90, c="#2ecc71", zorder=4, marker="o", label="start")
# 축 범위는 카메라 기준 (점군 outlier 무시하고 실제 동선에 맞춤)
ca, cb = cams[:, a], cams[:, b]
pad = max(ca.max() - ca.min(), cb.max() - cb.min()) * 0.5 + 0.5
ax.set_xlim(ca.min() - pad, ca.max() + pad); ax.set_ylim(cb.min() - pad, cb.max() + pad)
ax.set_title(f"{title} — {len(cams)} cameras (top view)")
ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="upper right")
plt.savefig(out, dpi=100, bbox_inches="tight")
print(f"saved {out}  ({len(cams)} cams, {len(pts)} pts)")
