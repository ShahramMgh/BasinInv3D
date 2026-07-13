#!/usr/bin/env python3
"""Write an imaginary microtremor field campaign to disk.

The campaign folder mimics real field data delivery (stations.csv plus raw
3-component records as .npz/.csv and a few pre-processed .hv curves) and can
be loaded straight into the Microtremor Studio's Field-assistant mode, or by
any code via basininv.fieldio.  A truth.npz is included so a validation run
can be scored — the inversion itself never reads it.

Usage:
    python3 phase2_toolbox/scripts/make_field_demo.py [folder] [--seed 2]
        [--layers 3] [--side 5] [--noise 5]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from basininv.fieldio import make_demo_campaign


def main():
    ap = argparse.ArgumentParser()
    default_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "outputs", "field_demo")
    ap.add_argument("folder", nargs="?", default=default_folder)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--side", type=int, default=5,
                    help="stations per side (side^2 total)")
    ap.add_argument("--noise", type=float, default=5.0,
                    help="noise %% on the pre-processed .hv stations")
    args = ap.parse_args()
    info = make_demo_campaign(args.folder, seed=args.seed,
                              n_layers=args.layers, n_side=args.side,
                              noise_pct=args.noise)
    kinds = {k: info["kinds"].count(k) for k in set(info["kinds"])}
    print(f"campaign written to {info['folder']}")
    print(f"  {info['n_stations']} stations: {kinds}")
    print(f"  hidden truth: Vs {info['vs_true']}, "
          f"max bedrock depth {info['max_depth']:.0f} m")
    print("open the field dashboard (python3 phase2_toolbox/webapp_field/app.py), "
          "scan this folder and press Run.")


if __name__ == "__main__":
    main()
