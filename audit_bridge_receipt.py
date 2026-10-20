"""Portable post-lock receipt audit with explicitly bounded bridge roundoff.

The producing source and original JSON are never changed. A deep audit copy can
normalize bridge entries by at most eight floating-point ULPs to account for
cross-runner NumPy exp rounding; every normalization is reported. All original
source-lock, physics, inputs, costs, endpoint and summary checks then run.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import bridge_robustness as br
from phase_bridge_gate import PhaseMedium

MAX_ULPS = 8


def audit_portable(receipt):
    normalized = copy.deepcopy(receipt)
    changed = 0
    max_abs = 0.
    max_ulps = 0.
    blocks = br.load_protocol()["blocks"]
    for device in normalized["devices"]:
        box = PhaseMedium(blocks=blocks, seed=device["seed"])
        mask = box.owner == -1
        expected = box.initial[mask]
        spacing = np.abs(np.spacing(expected))
        for name, values in device["materials"].items():
            state = np.asarray(values, dtype=float)
            if state.shape != box.initial.shape or not np.all(np.isfinite(state)):
                raise ValueError("invalid material in portable audit")
            difference = np.abs(state[mask]-expected)
            ulps = difference/spacing
            if np.any(ulps > MAX_ULPS):
                raise ValueError("bridge change exceeds permitted roundoff")
            changed += int(np.count_nonzero(difference))
            max_abs = max(max_abs, float(np.max(difference)))
            max_ulps = max(max_ulps, float(np.max(ulps)))
            state[mask] = expected
            device["materials"][name] = state.tolist()
    result = br.audit_receipt(normalized)
    result["roundoff_normalization"] = {
        "changed_bridge_values": changed,
        "max_absolute_difference": max_abs,
        "max_ulps": max_ulps,
        "limit_ulps": MAX_ULPS,
        "original_receipt_mutated": False,
        "scope": "Only immutable bridge entries in a deep audit copy; all other original audit checks retained.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt",type=Path)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    audit = audit_portable(json.loads(args.receipt.read_text(encoding="utf-8")))
    text = json.dumps(audit,indent=2)+"\n"
    if args.out:
        if args.out.exists():
            raise ValueError("refusing to overwrite audit record")
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(text,encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
