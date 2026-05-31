#!/usr/bin/env python3
import argparse
import csv
import math
import os
from typing import Tuple

import numpy as np


def huber_irls_fit(x: np.ndarray, y: np.ndarray, delta: float = 1.5, iters: int = 20) -> np.ndarray:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(iters):
        r = y - x @ beta
        scale = 1.4826 * np.median(np.abs(r)) + 1e-9
        threshold = delta * scale
        w = np.ones_like(r)
        mask = np.abs(r) > threshold
        w[mask] = threshold / (np.abs(r[mask]) + 1e-9)
        w = np.clip(w, 1e-3, 1.0)
        wx = x * w[:, None]
        wy = y * w
        beta = np.linalg.lstsq(wx, wy, rcond=None)[0]
    return beta


def build_design(inv_h: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(inv_h), inv_h, inv_h * inv_h], axis=1)


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def load_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = []
    bx = []
    by = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hv = float(row["h_px"])
                bxx = float(row["bias_x"])
                byy = float(row["bias_y"])
            except (KeyError, ValueError):
                continue
            if not math.isfinite(hv) or not math.isfinite(bxx) or not math.isfinite(byy):
                continue
            h.append(hv)
            bx.append(bxx)
            by.append(byy)
    if not h:
        raise RuntimeError(f"no valid samples in {path}")
    return np.array(h), np.array(bx), np.array(by)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit laser-camera bias from samples csv")
    parser.add_argument("--input", required=True, help="input csv path")
    parser.add_argument("--output", required=True, help="output yaml path")
    parser.add_argument("--h-min", type=float, default=4.0)
    parser.add_argument("--h-max", type=float, default=300.0)
    parser.add_argument("--delta", type=float, default=1.5, help="Huber delta multiplier")
    args = parser.parse_args()

    h, bx, by = load_csv(args.input)

    mask = (h >= args.h_min) & (h <= args.h_max)
    h = h[mask]
    bx = bx[mask]
    by = by[mask]
    if h.size < 12:
        raise RuntimeError(f"not enough filtered samples: {h.size} (<12)")

    inv_h = 1.0 / np.maximum(h, 1e-6)
    x = build_design(inv_h)

    coeff_x = huber_irls_fit(x, bx, delta=args.delta)
    coeff_y = huber_irls_fit(x, by, delta=args.delta)

    pred_x = x @ coeff_x
    pred_y = x @ coeff_y

    rmse_x = rmse(bx, pred_x)
    rmse_y = rmse(by, pred_y)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "w", newline="") as f:
        f.write("laser_camera_calibrate:\n")
        f.write("  enable: true\n")
        f.write("  apply_bias: true\n")
        f.write("  enable_sampling: false\n")
        f.write("  auto_detect_laser: true\n")
        f.write(f"  h_px_min: {float(np.min(h)):.6f}\n")
        f.write(f"  h_px_max: {float(np.max(h)):.6f}\n")
        f.write(f"  inv_h_min: {float(np.min(inv_h)):.9f}\n")
        f.write(f"  inv_h_max: {float(np.max(inv_h)):.9f}\n")
        f.write(
            "  coeff_x: [{:.12g}, {:.12g}, {:.12g}]\n".format(
                float(coeff_x[0]), float(coeff_x[1]), float(coeff_x[2])
            )
        )
        f.write(
            "  coeff_y: [{:.12g}, {:.12g}, {:.12g}]\n".format(
                float(coeff_y[0]), float(coeff_y[1]), float(coeff_y[2])
            )
        )
        f.write(f"  fit_sample_count: {int(h.size)}\n")
        f.write(f"  fit_rmse_x: {rmse_x:.6f}\n")
        f.write(f"  fit_rmse_y: {rmse_y:.6f}\n")

    print("fit done")
    print(f"samples={h.size}")
    print(f"rmse_x={rmse_x:.4f}px rmse_y={rmse_y:.4f}px")
    print(f"coeff_x={coeff_x}")
    print(f"coeff_y={coeff_y}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
