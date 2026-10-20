"""Retrain the locked bridge experiment; preserve the original receipt.

Run from an audit branch, passing a separate clean checkout of the source lock.
Only transparent call counters wrap the original writer and linear solver.
Numerical tolerances below were fixed before inspecting rerun differences.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
import numpy as np

LOCK = "7cb745c7d0b197742616401a5d1838d13f19854e"
WRITERS = ("standard", "noise_aware")
COUNTS = ("read_calls", "forward_solves", "adjoint_solves", "linear_update_solves", "bridge_writes")
TOLERANCES = {
    "conductances": {"rtol":1e-9, "atol":1e-11},
    "losses": {"rtol":1e-8, "atol":1e-15},
    "inputs": {"rtol":1e-12, "atol":1e-14},
}


def fit_counted(fit, box, targets, baseline, factors, cfg):
    """Count actual calls without changing arguments, return values or arithmetic."""
    counts = {"read_calls":0, "node_system_solves":0, "update_solves":0}
    original_read, original_solve = box.read, np.linalg.solve
    in_read = False

    def read(*args, **kwargs):
        nonlocal in_read
        counts["read_calls"] += 1
        in_read = True
        try:
            return original_read(*args, **kwargs)
        finally:
            in_read = False

    def solve(*args, **kwargs):
        counts["node_system_solves" if in_read else "update_solves"] += 1
        return original_solve(*args, **kwargs)

    box.read, np.linalg.solve = read, solve
    try:
        state, info = fit(box, targets, baseline, factors, cfg)
    finally:
        box.read, np.linalg.solve = original_read, original_solve
    return state, info, counts


def numeric_comparison(saved, rerun, kind, label):
    a, b = np.asarray(saved, dtype=float), np.asarray(rerun, dtype=float)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("invalid numerical field: " + label)
    if not np.allclose(a, b, **TOLERANCES[kind]):
        raise ValueError("training reproduction mismatch: " + label)
    delta = np.abs(a-b)
    return {"values_compared":int(a.size), "max_absolute_difference":float(delta.max()),
            "max_relative_difference":float(np.max(delta/np.maximum(np.abs(a), np.finfo(float).tiny)))}


def compare_training(saved, rerun, observed):
    """Compare every conductance and training loss, plus exact counted budgets."""
    for key in ("experiment", "mode", "held_out", "scientific_claims_evaluated",
                "lock_commit", "protocol", "effective_config", "source_sha256"):
        if saved[key] != rerun[key]:
            raise ValueError("training provenance/config mismatch: " + key)
    seeds = [d["seed"] for d in saved["devices"]]
    if seeds != [d["seed"] for d in rerun["devices"]] or seeds != saved["effective_config"]["seeds"][saved["mode"]]:
        raise ValueError("training seed mismatch")
    if set(observed) != {str(seed) for seed in seeds}:
        raise ValueError("instrumentation device mismatch")
    comparisons = []
    for a, b in zip(saved["devices"], rerun["devices"]):
        seed = a["seed"]
        for key in ("teacher_signs", "targets", "baseline", "training_noise_factors"):
            numeric_comparison(a[key], b[key], "inputs", f"{seed}.{key}")
        if set(a["materials"]) != set(WRITERS) or set(b["materials"]) != set(WRITERS):
            raise ValueError("training writer mismatch")
        for writer in WRITERS:
            ai, bi = a["training"][writer], b["training"][writer]
            count = observed[str(seed)][writer]
            if any(type(ai[k]) is not int or ai[k] != bi[k] for k in COUNTS):
                raise ValueError("recorded solve count mismatch")
            expected = {
                "read_calls":ai["read_calls"],
                "node_system_solves":ai["forward_solves"]+ai["adjoint_solves"],
                "update_solves":ai["linear_update_solves"],
            }
            if count != expected:
                raise ValueError("observed solve count mismatch")
            history = numeric_comparison(ai["loss_before_each_update"], bi["loss_before_each_update"], "losses", f"{seed}.{writer}.history")
            endpoints = numeric_comparison([ai["initial_loss"], ai["final_loss"]], [bi["initial_loss"], bi["final_loss"]], "losses", f"{seed}.{writer}.loss")
            fields = numeric_comparison(a["materials"][writer], b["materials"][writer], "conductances", f"{seed}.{writer}.material")
            comparisons.append({"seed":seed, "writer":writer,
                "conductance_comparison":fields, "loss_history_comparison":history,
                "endpoint_loss_comparison":endpoints, "observed_calls":count,
                "recorded_counts":{k:ai[k] for k in COUNTS},
                "rerun_conductances":b["materials"][writer],
                "rerun_training":bi})
    return {"passed":True, "devices_retrained":len(seeds), "writers_retrained":len(comparisons),
            "intermediate_losses_compared":sum(x["loss_history_comparison"]["values_compared"] for x in comparisons),
            "tolerances":TOLERANCES, "comparisons":comparisons}


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def worker(source, out):
    # Source directory goes ahead of the audit branch in the fresh interpreter.
    sys.path.insert(0, str(source))
    import bridge_robustness as br
    if br.ROOT.resolve() != source:
        raise ValueError("wrong source module loaded")
    br.lock_source(LOCK, br.load_protocol())
    counts = {}
    fit = br.fit_material

    def instrumented(box, targets, baseline, factors, cfg):
        # run_experiment creates one box per protocol seed and invokes writers in order.
        index = sum(len(v) for v in counts.values())
        seed = br.load_protocol()["seeds"]["full"][index//2]
        writer = WRITERS[index % 2]
        state, info, observed = fit_counted(fit, box, targets, baseline, factors, cfg)
        counts.setdefault(str(seed), {})[writer] = observed
        return state, info

    br.fit_material = instrumented
    try:
        rerun = br.run_experiment("full", LOCK)
    finally:
        br.fit_material = fit
    br.lock_source(LOCK, br.load_protocol())
    write_new(out, {"rerun":rerun, "observed":counts,
        "environment":{"python":platform.python_version(), "numpy":np.__version__,
            "platform":platform.platform(), "machine":platform.machine(),
            "thread_environment":{k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "numpy_build":np.__config__.CONFIG}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.out.exists():
        raise ValueError("refusing to overwrite an audit")
    if args.worker:
        worker(source, args.out)
        return
    if args.receipt is None:
        parser.error("--receipt is required")
    receipt_bytes = args.receipt.read_bytes()
    saved = json.loads(receipt_bytes)
    if saved["lock_commit"] != LOCK or saved["mode"] != "full":
        raise ValueError("expected the original full locked receipt")
    env = os.environ | {"OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}
    with tempfile.TemporaryDirectory(prefix="bridge-retrain-") as directory:
        temp = Path(directory)/"rerun.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
            "--source", str(source), "--out", str(temp)], cwd=source, env=env, check=True)
        payload_bytes = temp.read_bytes()
        payload = json.loads(payload_bytes)
    result = compare_training(saved, payload["rerun"], payload["observed"])
    import bridge_robustness as br
    from audit_bridge_receipt import audit_portable
    result["original_endpoint_audit"] = audit_portable(saved)
    result["rerun_endpoint_audit"] = audit_portable(payload["rerun"])
    br._close(saved["summary"], payload["rerun"]["summary"], "retrained full summary")
    if receipt_bytes != args.receipt.read_bytes():
        raise ValueError("original receipt changed during audit")
    result.update({"audit_kind":"independent locked training reproduction",
        "date_utc":datetime.now(timezone.utc).isoformat(), "lock_commit":LOCK,
        "original_receipt_sha256":hashlib.sha256(receipt_bytes).hexdigest(),
        "auditor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256":payload["rerun"]["source_sha256"],
        "environment":payload["environment"], "original_receipt_unchanged":True,
        "training_source_modified":False,
        "instrumentation":"Transparent wrappers count actual read calls, node linear solves and update linear solves during each original writer; no numerical arguments or results changed.",
        "summary_matches":True,
        "original_primary_target_errors":saved["summary"]["conditions"]["independent:0.6"]["mean_target_error"],
        "rerun_primary_target_errors":payload["rerun"]["summary"]["conditions"]["independent:0.6"]["mean_target_error"],
        "limitations":["Same implementation rerun in a fresh interpreter, not an independent optimizer reimplementation.",
            "Actual node-system solves are counted jointly; the original source executes one forward and one adjoint solve per read.",
            "Tolerances allow small platform/BLAS roundoff; this audit is not a new held-out experiment."]})
    write_new(args.out, result)
    print(json.dumps({k:result[k] for k in ("passed", "devices_retrained", "writers_retrained", "intermediate_losses_compared", "summary_matches", "original_receipt_unchanged")}))


if __name__ == "__main__":
    main()
