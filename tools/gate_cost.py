"""Quantify the computational cost of the adaptive gating mechanisms.

Reports, for a given (L, H, C) configuration:
  - parameter counts of the fusion gate (gate_base), the CD gate (gate_cd),
    and their share of the full FreqNetCD model;
  - forward latency with gating enabled vs. fixed equal-weight fusion,
    measured over repeated batches.

Run from the TSLib root (needs models/FreqNetCD.py importable):
  python tools/gate_cost.py --channels 862 --batch 32
"""
import argparse
import sys
import time
import types

import torch

sys.path.insert(0, ".")
from models.FreqNetCD import Model  # noqa: E402


def make_cfg(seq_len=96, pred_len=96, channels=7, adaptive=True):
    cfg = types.SimpleNamespace()
    cfg.task_name = "long_term_forecast"
    cfg.seq_len = seq_len
    cfg.pred_len = pred_len
    cfg.enc_in = channels
    cfg.freqnet_adaptive_fusion = adaptive
    return cfg


def count(m):
    return sum(p.numel() for p in m.parameters())


def latency(model, x, device, iters=50, warmup=10):
    model = model.to(device).eval()
    x = x.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x, None, None, None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x, None, None, None)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--channels", type=int, default=862)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = Model(make_cfg(args.seq_len, args.pred_len, args.channels, True))
    total = count(model)
    gate_base = count(model.gate_base)
    gate_cd = count(model.gate_cd) if hasattr(model, "gate_cd") else 0
    print(f"total params          : {total:,}")
    print(f"fusion gate params    : {gate_base:,} ({100*gate_base/total:.2f}%)")
    print(f"CD gate params        : {gate_cd:,} ({100*gate_cd/total:.2f}%)")
    print(f"gates combined        : {gate_base+gate_cd:,} "
          f"({100*(gate_base+gate_cd)/total:.2f}%)")

    x = torch.randn(args.batch, args.seq_len, args.channels)
    lat_adaptive = latency(model, x, device)
    model_fixed = Model(make_cfg(args.seq_len, args.pred_len, args.channels, False))
    lat_fixed = latency(model_fixed, x, device)
    print(f"latency adaptive gate : {lat_adaptive:.3f} ms/batch (B={args.batch})")
    print(f"latency fixed 0.5/0.5 : {lat_fixed:.3f} ms/batch")
    print(f"gating overhead       : {lat_adaptive-lat_fixed:+.3f} ms "
          f"({100*(lat_adaptive-lat_fixed)/lat_fixed:+.2f}%)")


if __name__ == "__main__":
    main()
