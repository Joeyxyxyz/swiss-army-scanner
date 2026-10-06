"""Command-line interface."""

from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict, is_dataclass
import numpy as np

from .core import SensorRig, SensorMode


def _to_jsonable(obj):
    if is_dataclass(obj):
        d = asdict(obj)
    elif isinstance(obj, dict):
        d = obj
    else:
        return repr(obj)
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = {"shape": list(v.shape), "dtype": str(v.dtype),
                      "mean": float(v.mean()), "std": float(v.std())}
        else:
            out[k] = v
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="swiss-army-sensor",
        description="Turn your DLP/DMD into any sensor.")
    p.add_argument("--mock", action="store_true", default=True,
                   help="Use mock hardware (default).")
    p.add_argument("--real", action="store_true",
                   help="Use real DLP4500 + FLIR camera.")
    p.add_argument("--scene", default="plane",
                   choices=["plane", "sphere", "step", "qr"],
                   help="Synthetic scene for mock mode.")
    p.add_argument("--mode", required=True,
                   choices=[m.value for m in SensorMode])
    p.add_argument("--params", default="{}",
                   help='JSON dict of mode parameters')

    args = p.parse_args(argv)
    mock = not args.real
    params = json.loads(args.params)

    rig = SensorRig.create(mock=mock, scene=args.scene)
    try:
        rig.set_mode(SensorMode(args.mode), **params)
        result = rig.read()
        print(json.dumps(_to_jsonable(result), indent=2, default=str))
    finally:
        rig.close()


if __name__ == "__main__":
    main()
