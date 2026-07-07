#!/usr/bin/env python3
"""Generate bottom-band masks to exclude the camera operator from 360 training.

On an equirectangular capture the person holding the camera is fixed at the
nadir (bottom of the frame). A single bottom band, replicated per view, removes
them from supervision. Masked (black) pixels are excluded from the loss by
train.py; white pixels are trained normally.

Usage:
    python tools/make_mask.py data/your_data --frac 0.30

Writes data/your_data/masks/<image_name>.png (one per image, same size).
Train with:  train.py -s data/your_data --panorama --eval --masks data/your_data/masks
"""
import argparse
import os

import numpy as np
from PIL import Image

EXTS = (".jpg", ".jpeg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="dataset dir containing the images folder")
    ap.add_argument("--images", default="images", help="images subfolder name")
    ap.add_argument("--frac", type=float, default=0.30,
                    help="bottom fraction to mask out (0-1). 0.30 covers most selfie-stick captures")
    args = ap.parse_args()

    img_dir = os.path.join(args.data_dir, args.images)
    mask_dir = os.path.join(args.data_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(EXTS))
    if not names:
        raise SystemExit(f"no images found in {img_dir}")

    w, h = Image.open(os.path.join(img_dir, names[0])).size
    cut = int(round(h * (1 - args.frac)))
    band = np.full((h, w), 255, np.uint8)
    band[cut:, :] = 0  # black = excluded from training
    tmpl = Image.fromarray(band, "L")

    for n in names:
        tmpl.save(os.path.join(mask_dir, n + ".png"))
    print(f"{len(names)} masks written to {mask_dir}  (bottom {args.frac*100:.0f}%, cut y={cut}/{h})")


if __name__ == "__main__":
    main()
