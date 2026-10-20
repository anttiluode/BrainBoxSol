"""Noise-aware conductance writing versus interface repair.

Supervised, simultaneous Gauss-Newton regression on PhaseMedium. No biological
locality or invariant-subspace claim. See protocols/bridge_robustness.json.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from phase_bridge_gate import PhaseMedium, teacher, homeostat

ROOT = Path(__file__).resolve().parent
PROTOCOL = ROOT / "protocols/bridge_robustness.json"
ARMS = ("standard", "noise_aware", "standard_repaired", "combined")
SOURCE_FILES = ("bridge_robustness.py", "phase_bridge_gate.py", "brainboxsol.py",
                "protocols/bridge_robustness.json")


def load_protocol():
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def rng_for(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([20261020, int(seed), int(stream)]))


def noise_factors(box, rng, family, sd, count):
    """Positive multiplicative hardware factors; each draw is shared by all arms."""
    if family not in ("mixed", "independent", "coherent", "alternating", "interior"):
        raise ValueError("unknown drift family")
    if not np.isfinite(sd) or sd < 0 or count < 1:
        raise ValueError("invalid noise scale/count")
    factors = np.ones((count, len(box.initial)))
    if sd == 0:
        return factors
    if family == "interior":
        mask = box.owner >= 0
        factors[:, mask] = np.exp(sd * rng.standard_normal((count, int(mask.sum()))))
        return factors
    mask = box.interface >= 0
    interface = box.interface[mask]
    independent = rng.standard_normal((count, int(mask.sum())))
    coherent = rng.standard_normal((count, box.blocks-1))[:, interface]
    if family == "mixed":
        noise = (independent + coherent) / np.sqrt(2.)
    elif family == "independent":
        noise = independent
    elif family == "coherent":
        noise = coherent
    else:
        amplitude = rng.standard_normal((count, 1))
        noise = amplitude * np.where(interface % 2 == 0, 1., -1.)[None, :]
    factors[:, mask] = np.exp(sd * noise)
    return factors


def residual_jacobian(box, state, targets, baseline, factors):
    """Objective residuals and exact derivative w.r.t. undrifted writable state.

    Include the multiplicative chain rule: d y(state*f)/d state = dy/dg * f.
    Rows are scaled so squared Euclidean norm is mean normalized squared error.
    """
    state = np.asarray(state, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if factors.ndim != 2 or factors.shape[1] != len(state):
        raise ValueError("noise bank shape mismatch")
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        raise ValueError("invalid hardware factors")
    if np.any(np.asarray(baseline) <= 0):
        raise ValueError("positive normalization required")
    writable = box.owner >= 0
    scale = np.sqrt(len(factors)*box.blocks)
    residuals, rows = [], []
    for f in factors:
        g = state*f
        for k in range(box.blocks):
            value, sensitivity = box.read(g, k)
            residuals.append((value-targets[k])/baseline[k]/scale)
            rows.append(sensitivity[writable]*f[writable]/baseline[k]/scale)
    return np.array(residuals), np.array(rows)


def fit_material(box, targets, baseline, factors, cfg):
    """Fixed-budget damped Gauss-Newton; bridges frozen, only interior writable."""
    if len(factors) != cfg["worlds"] or cfg["steps"] < 1:
        raise ValueError("training world count/steps mismatch")
    state = box.initial.copy()
    writable = box.owner >= 0
    losses = []
    for _ in range(cfg["steps"]):
        residual, jac = residual_jacobian(box, state, targets, baseline, factors)
        losses.append(float(residual @ residual))
        gram = jac.T @ jac
        damping = cfg["ridge"] * max(float(np.max(np.diag(gram))), 1e-20)
        update = np.linalg.solve(gram + damping*np.eye(gram.shape[0]), -jac.T @ residual)
        maximum = float(np.max(np.abs(update)))
        if maximum > cfg["max_step"]:
            update *= cfg["max_step"]/maximum
        state[writable] = np.clip(state[writable] + cfg["rate"]*update,
                                  *cfg["conductance_bounds"])
    residual, _ = residual_jacobian(box, state, targets, baseline, factors)
    calls = (cfg["steps"]+1)*len(factors)*box.blocks
    return state, {
        "initial_loss": losses[0], "final_loss": float(residual @ residual),
        "loss_before_each_update": losses, "read_calls": calls,
        "forward_solves": calls, "adjoint_solves": calls,
        "linear_update_solves": cfg["steps"],
        "bridge_writes": 0,
    }


def normalized_error(values, targets, baseline):
    return float(np.mean(np.abs(np.asarray(values)-targets)/baseline))


def effective_config(protocol, mode):
    if mode not in ("smoke", "development", "full"):
        raise ValueError("unknown mode")
    cfg = copy.deepcopy(protocol)
    if mode == "smoke":
        cfg["training"].update(steps=4, worlds=2)
        cfg["evaluation"].update(strengths=[.6], families=["coherent", "interior"], draws=2)
    return cfg


def lock_source(lock_commit, protocol):
    if not lock_commit:
        raise ValueError("full mode requires an exact clean lock commit")
    if not protocol.get("locked"):
        raise ValueError("protocol is not locked")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    head = git("rev-parse", "HEAD")
    expected = git("rev-parse", lock_commit + "^{commit}")
    if head != expected or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("full mode requires a clean checkout at the lock commit")
    for path in SOURCE_FILES:
        original = subprocess.check_output(["git", "show", expected+":"+path], cwd=ROOT)
        if original != (ROOT/path).read_bytes():
            raise ValueError("lock source mismatch: "+path)
    return expected


def evaluate_device(box, states, targets, baseline, cfg, seed):
    """Common random numbers, no targets or readout used by interface repair."""
    clean_values = {key: box.values(value) for key,value in states.items()}
    records = []
    ecfg = cfg["evaluation"]
    for fi, family in enumerate(ecfg["families"]):
        for si, strength in enumerate(ecfg["strengths"]):
            factors = noise_factors(box, rng_for(seed, 1000+100*fi+si),
                                    family, strength, ecfg["draws"])
            errors = {arm: [] for arm in ARMS}
            changes = {arm: [] for arm in ARMS}
            frozen = []
            for f in factors:
                frozen.append(normalized_error(box.values(box.initial*f), targets, baseline))
                for writer in ("standard", "noise_aware"):
                    trained = states[writer]
                    damaged = trained*f
                    repaired = homeostat(box, damaged, trained, "interface")
                    for arm, material in (
                        (writer, damaged),
                        ("standard_repaired" if writer == "standard" else "combined", repaired),
                    ):
                        values = box.values(material)
                        errors[arm].append(normalized_error(values, targets, baseline))
                        changes[arm].append(normalized_error(values, clean_values[writer], baseline))
            records.append({
                "family": family, "strength": float(strength),
                "factors": factors.tolist(), "target_errors": errors,
                "trained_response_changes": changes, "frozen_target_errors": frozen,
            })
    return records


def summarize(devices, cfg, held_out):
    conditions = {}
    clean = {
        writer: float(np.mean([d["clean_errors"][writer] for d in devices]))
        for writer in ("standard", "noise_aware")
    }
    for family in cfg["evaluation"]["families"]:
        for strength in cfg["evaluation"]["strengths"]:
            records = [next(r for r in d["evaluation"]
                            if r["family"] == family and r["strength"] == strength)
                       for d in devices]
            means = {arm: [float(np.mean(r["target_errors"][arm])) for r in records]
                     for arm in ARMS}
            conditions[f"{family}:{strength:g}"] = {
                "mean_target_error": {arm: float(np.mean(x)) for arm,x in means.items()},
                "per_device_target_error": means,
                "mean_trained_response_change": {
                    arm: float(np.mean([np.mean(r["trained_response_changes"][arm])
                                        for r in records])) for arm in ARMS
                },
                "mean_frozen_target_error": float(np.mean(
                    [np.mean(r["frozen_target_errors"]) for r in records])),
            }
    ecfg = cfg["evaluation"]
    primary = conditions[f'{ecfg["primary_family"]}:{ecfg["primary_strength"]:g}']
    claim_rules = cfg["claims"]
    claims = {}
    for label,candidate,control in (
        ("noise_aware_vs_standard", "noise_aware", "standard"),
        ("combined_vs_repair", "combined", "standard_repaired"),
    ):
        c = np.array(primary["per_device_target_error"][control])
        t = np.array(primary["per_device_target_error"][candidate])
        difference = c-t
        rng = rng_for(0, 8000)
        bootstrap = difference[rng.integers(0, len(devices), (2000,len(devices)))].mean(1)
        reduction = float((c.mean()-t.mean())/max(float(c.mean()), 1e-12))
        positive = float(np.mean(difference > 1e-12))
        penalty = clean["noise_aware"]-clean["standard"]
        checks = {
            "relative_error_reduction": reduction >= claim_rules["min_relative_error_reduction"],
            "device_consistency": positive >= claim_rules["min_positive_device_fraction"],
            "clean_cost": penalty <= claim_rules["max_clean_error_penalty"],
        }
        claims[label] = {
            "candidate": candidate, "control": control,
            "relative_error_reduction": reduction, "positive_device_fraction": positive,
            "clean_error_penalty": penalty,
            "mean_paired_error_reduction": float(difference.mean()),
            "paired_device_bootstrap_95": np.quantile(bootstrap,[.025,.975]).tolist(),
            "criteria": checks, "passed": bool(all(checks.values())) if held_out else None,
        }
    return {"clean_mean_target_error": clean, "conditions": conditions, "claims": claims,
            "replication_unit": "device; ports and drift draws are averaged within device"}


def run_experiment(mode="development", lock_commit=None):
    protocol = load_protocol()
    locked_sha = lock_source(lock_commit, protocol) if mode == "full" else None
    cfg = effective_config(protocol, mode)
    devices = []
    start = time.perf_counter()
    for seed in cfg["seeds"][mode]:
        box = PhaseMedium(blocks=cfg["blocks"], seed=seed)
        signs = np.where(rng_for(seed,1).random(box.blocks)<.5, 1, -1)
        targets = box.values(teacher(box, signs))
        baseline = box.values(box.initial)
        bank = noise_factors(box, rng_for(seed,2), cfg["training"]["family"],
                             cfg["training"]["noise_sd"], cfg["training"]["worlds"])
        states, training = {}, {}
        for writer in ("standard", "noise_aware"):
            factors = np.ones_like(bank) if writer == "standard" else bank
            states[writer], training[writer] = fit_material(
                box, targets, baseline, factors, cfg["training"])
        devices.append({
            "seed": seed, "teacher_signs": signs.tolist(),
            "targets": targets.tolist(), "baseline": baseline.tolist(),
            "training_noise_factors": bank.tolist(),
            "materials": {k:v.tolist() for k,v in states.items()},
            "training": training,
            "clean_errors": {k:normalized_error(box.values(v),targets,baseline)
                             for k,v in states.items()},
            "frozen_clean_error": normalized_error(baseline, targets, baseline),
            "evaluation": evaluate_device(box, states, targets, baseline, cfg, seed),
        })
        print(f"device {seed}: clean target error "+
              json.dumps(devices[-1]["clean_errors"]), flush=True)
    held_out = mode == "full"
    return {
        "experiment": "bridge_robustness", "mode": mode, "held_out": held_out,
        "scientific_claims_evaluated": held_out, "lock_commit": locked_sha,
        "protocol": protocol, "effective_config": cfg,
        "source_sha256": {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                          for p in SOURCE_FILES},
        "devices": devices, "summary": summarize(devices,cfg,held_out),
        "elapsed_seconds_before_audit": time.perf_counter()-start,
        "limitations": [
            "Known ports and supervised targets, global damped Gauss-Newton updates.",
            "Noise training uses a fixed eight-world bank; no invariant-subspace theorem.",
            "Interface repair stores a pre-drift total per interface and senses its current sum.",
            "Matched read/adjoint/update counts do not establish digital efficiency.",
            "Interior drift is a diagnostic beyond the bridge repair mechanism's scope.",
            "Bootstrap intervals resample devices, not individual ports or drift draws.",
            "A six-value digital target table would solve the fixed-port task exactly and cheaply.",
        ],
    }


def _close(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("receipt keys differ: "+label)
        for key in expected:
            _close(actual[key],expected[key],label+"."+str(key))
    elif isinstance(expected, (list,tuple)):
        if not isinstance(actual, (list,tuple)) or len(actual) != len(expected):
            raise ValueError("receipt list differs: "+label)
        for i,(a,b) in enumerate(zip(actual,expected)):
            _close(a,b,label+"."+str(i))
    elif isinstance(expected, (int,float)) and not isinstance(expected,bool):
        if not np.isfinite(actual) or not np.isclose(actual,expected,rtol=1e-8,atol=1e-10):
            raise ValueError("receipt numeric mismatch: "+label)
    elif actual != expected:
        raise ValueError("receipt mismatch: "+label)


def audit_receipt(receipt):
    """Regenerate inputs and recompute every evaluation from stored materials."""
    mode = receipt["mode"]
    if receipt["protocol"] != load_protocol():
        raise ValueError("protocol differs")
    held_out = mode == "full"
    if receipt["held_out"] != held_out or receipt["scientific_claims_evaluated"] != held_out:
        raise ValueError("invalid held-out flag")
    if held_out and not receipt.get("lock_commit"):
        raise ValueError("missing lock provenance")
    cfg = effective_config(receipt["protocol"],mode)
    _close(receipt["effective_config"],cfg,"config")
    _close([d["seed"] for d in receipt["devices"]],cfg["seeds"][mode],"device seeds")
    evaluations = 0
    for d in receipt["devices"]:
        seed = d["seed"]
        box = PhaseMedium(blocks=cfg["blocks"],seed=seed)
        signs = np.where(rng_for(seed,1).random(box.blocks)<.5, 1, -1)
        targets = box.values(teacher(box,signs))
        baseline = box.values(box.initial)
        _close(d["teacher_signs"],signs.tolist(),"teacher signs")
        _close(d["targets"],targets.tolist(),"targets")
        _close(d["baseline"],baseline.tolist(),"baseline")
        bank = noise_factors(box,rng_for(seed,2),cfg["training"]["family"],
                             cfg["training"]["noise_sd"],cfg["training"]["worlds"])
        _close(d["training_noise_factors"],bank.tolist(),"training bank")
        states = {k:np.array(v) for k,v in d["materials"].items()}
        if set(states) != {"standard","noise_aware"}:
            raise ValueError("material arm mismatch")
        for writer,state in states.items():
            if state.shape != box.initial.shape or not np.all(np.isfinite(state)):
                raise ValueError("invalid material")
            mask = box.owner >= 0
            lo,hi = cfg["training"]["conductance_bounds"]
            if np.any(state[mask] < lo) or np.any(state[mask] > hi):
                raise ValueError("material bounds violated")
            np.testing.assert_array_equal(state[~mask],box.initial[~mask])
            _close(d["clean_errors"][writer],
                   normalized_error(box.values(state),targets,baseline),"clean errors")
            factors = np.ones_like(bank) if writer == "standard" else bank
            final = residual_jacobian(box,state,targets,baseline,factors)[0]
            initial = residual_jacobian(box,box.initial,targets,baseline,factors)[0]
            _close(d["training"][writer]["final_loss"],float(final@final),"final loss")
            _close(d["training"][writer]["initial_loss"],float(initial@initial),"initial loss")
            calls = (cfg["training"]["steps"]+1)*cfg["training"]["worlds"]*box.blocks
            for key in ("read_calls","forward_solves","adjoint_solves"):
                _close(d["training"][writer][key],calls,"training counts")
        _close(d["frozen_clean_error"],normalized_error(baseline,targets,baseline),"frozen clean")
        expected = evaluate_device(box,states,targets,baseline,cfg,seed)
        _close(d["evaluation"],expected,"evaluation")
        evaluations += len(expected)*cfg["evaluation"]["draws"]*len(ARMS)
    _close(receipt["summary"],summarize(receipt["devices"],cfg,held_out),"summary")
    expected_hashes = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES}
    _close(receipt["source_sha256"],expected_hashes,"source hashes")
    return {"passed": True, "devices_checked": len(receipt["devices"]),
            "paired_arm_evaluations_recomputed": evaluations,
            "training_rerun": False,
            "scope": "Regenerated inputs, checked stored materials/fit objectives/costs, recomputed all evaluation and summary; did not retrain."}


def render_report(receipt):
    rows = []
    for condition,group in receipt["summary"]["conditions"].items():
        cells = "".join(f"<td>{group['mean_target_error'][arm]:.6g}</td>" for arm in ARMS)
        rows.append("<tr><th>"+html.escape(condition)+"</th>"+cells+"</tr>")
    claims = html.escape(json.dumps(receipt["summary"]["claims"],indent=2))
    limits = "".join("<li>"+html.escape(x)+"</li>" for x in receipt["limitations"])
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>BrainBoxSol: bridge robustness</title>'
        '<style>body{font:17px/1.6 system-ui;max-width:1050px;margin:40px auto;padding:0 20px;'
        'background:#101722;color:#e8edf5}h1{line-height:1.2}table{border-collapse:collapse;width:100%}'
        'td,th{padding:10px;border-bottom:1px solid #3c4655;text-align:left}'
        'pre{overflow:auto;background:#192333;padding:18px}small{color:#b9c6d8}</style>'
        '<h1>Does noise-aware writing complement interface repair?</h1>'
        '<p>Mode: <b>'+html.escape(receipt["mode"])+'</b>. '
        'Mean absolute normalized target-response error; lower is better. These are regression errors, not classification accuracies.</p>'
        '<p>Clean error: '+html.escape(json.dumps(receipt["summary"]["clean_mean_target_error"]))+'</p>'
        '<div style="overflow:auto"><table><tr><th>Drift family / strength</th>'
        '<th>Standard</th><th>Noise-aware</th><th>Repair only</th><th>Combined</th></tr>'
        +"".join(rows)+'</table></div><h2>Predeclared claims</h2><pre>'+claims+
        '</pre><h2>Limits</h2><ul>'+limits+'</ul><small>Lock: '+
        html.escape(str(receipt["lock_commit"]))+
        '. Device-level paired evaluation; see adjacent JSON for full numerical receipt.</small></html>'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["smoke","development","full"],default="development")
    parser.add_argument("--lock-commit")
    parser.add_argument("--out",type=Path,default=Path("results/bridge_robustness/development.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite an existing result receipt")
    result = run_experiment(args.mode,args.lock_commit)
    result["audit"] = audit_receipt(result)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    args.out.with_suffix(".html").write_text(render_report(result),encoding="utf-8")
    print("BRIDGE_SUMMARY_BEGIN")
    print(json.dumps({"mode":result["mode"],"lock_commit":result["lock_commit"],
                      "source_sha256":result["source_sha256"],"summary":result["summary"],
                      "audit":result["audit"]},sort_keys=True))
    print("BRIDGE_SUMMARY_END")
    # Machine-readable complete receipt for remote-only runners; also uploaded as an artifact.
    print("BRIDGE_RECEIPT_BEGIN")
    print(json.dumps(result,separators=(",",":"),sort_keys=True))
    print("BRIDGE_RECEIPT_END")


if __name__ == "__main__":
    main()
