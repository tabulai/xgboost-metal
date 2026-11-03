#!/usr/bin/env python3
"""Serial fresh-process deep-tree experiment with pinned, independently selectable arms.

An arms JSON object maps names to {package_root, native_path, native_sha256,
params}. Optional source_root records tracked source provenance; optional
expected_gpu_tree (0/1 or a case-name mapping) asserts actual driver selection.
Relative arm paths resolve against the arms file's directory. Exported baseline
artifact paths remain absolute; preserve their referenced process directories.
Use --compare-baseline OLD_OUTPUT/baseline-artifacts.json to require exact saved
outputs from another run/binary. Every run requires a fresh output directory.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
from deep_tree_worker import canonical_model


HERE = Path(__file__).resolve().parent
CASES = {f"{density}-d{depth}-{driver}": dict(density=density, depth=depth, driver=driver)
         for depth in (6, 8, 10) for density in ("dense", "missing") for driver in ("gpu", "host")}
# Keep the original twelve cases as the default; deeper host coverage is opt-in.
DEFAULT_CASES = tuple(CASES)
CASES.update({f"{density}-d12-host": dict(density=density, depth=12, driver="host")
              for density in ("dense", "missing")})
REMOVE = {"PYTHONPATH", "PYTHONOPTIMIZE", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC",
          "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "VECLIB_MAXIMUM_THREADS",
          "METAL_DEVICE_WRAPPER_TYPE"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def array_record(value):
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                c_order_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def source_record(source):
    source = Path(source).resolve()
    command = ["git", "-C", str(source)]
    names = subprocess.check_output(command + ["ls-files", "-z"]).decode().split("\0")
    return dict(root=str(source), head=subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip(),
                status=subprocess.check_output(command + ["status", "--porcelain=v1", "--untracked-files=no"],
                                               text=True),
                diff_sha256=hashlib.sha256(subprocess.check_output(command + ["diff", "--binary", "HEAD"])).hexdigest(),
                files_sha256={name: sha(source / name) for name in sorted(names) if name})


def arm_snapshot(arm):
    package = Path(arm["package_root"]) / "xgboost"
    result = dict(native_sha256=sha(arm["native_path"]),
                  package_files_sha256={str(p.relative_to(package)): sha(p)
                      for p in sorted(package.rglob("*"))
                      if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts})
    if arm.get("source_root"):
        result["source"] = source_record(arm["source_root"])
    return result


def quick_pins(arms, worker, worker_sha):
    if sha(worker) != worker_sha:
        raise RuntimeError("Worker changed during experiment")
    for name, arm in arms.items():
        if sha(arm["native_path"]) != arm["native_sha256"]:
            raise RuntimeError(f"Native library changed: {name}")


def outputs(directory, prefix):
    records = {}
    for suffix, field in (("model.ubj", "model_ubj_sha256"), ("model.json", "model_json_sha256"),
                          ("model.canonical.json", "canonical_json_sha256")):
        records[field] = sha(directory / (prefix + "." + suffix))
    raw = (directory / (prefix + ".model.json")).read_bytes()
    canonical = (directory / (prefix + ".model.canonical.json")).read_bytes()
    if canonical_model(raw)[0] != canonical:
        raise RuntimeError("Saved canonical JSON does not normalize saved raw model")
    pred = np.load(directory / (prefix + ".predictions.npy"), allow_pickle=False)
    if not np.isfinite(pred).all():
        raise RuntimeError("Nonfinite saved predictions")
    records["prediction"] = array_record(pred)
    labels = np.load(directory / "query-label.npy", allow_pickle=False)
    if pred.shape != labels.shape:
        raise RuntimeError("Prediction/label shapes differ")
    records["query_label"] = array_record(labels)
    records["rmse"] = float(np.sqrt(np.mean((pred.astype(np.float64) - labels) ** 2)))
    return records


def audit(record, arm, worker_sha, args, exact, exported, external):
    directory = Path(record["directory"])
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "passed" or len(result["measurements"]) != args.repeats:
        raise RuntimeError("Worker failed or returned incomplete fits")
    ident = result["identity"]
    if (Path(ident["package"]).resolve() != Path(arm["package_root"]) / "xgboost/__init__.py"
            or Path(ident["native"]).resolve() != Path(arm["native_path"])
            or ident["native_sha256"] != arm["native_sha256"] or ident["worker_sha256"] != worker_sha):
        raise RuntimeError("Worker identity does not match pinned arm")
    if result["case"] != record["case"] or result["arm"] != record["arm"]:
        raise RuntimeError("Worker case/arm identity mismatch")
    if result["rounds"] != args.rounds or result["warmup_rounds"] != args.warmup_rounds:
        raise RuntimeError("Worker training length mismatch")
    params = dict(tree_method="hist", objective="reg:squarederror", max_depth=CASES[record["case"]]["depth"],
                  max_bin=args.bins, base_score=0, seed=271, eta=0.1, nthread=1,
                  metal_gpu_tree=int(CASES[record["case"]]["driver"] == "gpu"))
    params.update(arm["params"])
    if result["params"] != params:
        raise RuntimeError("Worker parameters differ from plan")
    fits = []
    for index, fit in enumerate(result["measurements"]):
        prefix = f"repeat-{index:02d}"
        if fit["repeat"] != index or not math.isfinite(fit["train_seconds"]) or fit["train_seconds"] <= 0:
            raise RuntimeError("Invalid timing receipt")
        if fit != json.loads((directory / (prefix + ".json")).read_text()):
            raise RuntimeError("Process and per-fit receipts disagree")
        saved = outputs(directory, prefix)
        for field in ("model_ubj_sha256", "model_json_sha256", "canonical_json_sha256", "prediction", "rmse"):
            if saved[field] != fit[field]:
                raise RuntimeError(f"Saved artifact/receipt mismatch: {field}")
        if saved["query_label"] != result["data"]["query_label"]:
            raise RuntimeError("Saved query labels do not match input identity")
        equality = {**saved, "data": result["data"], "rounds": result["rounds"], "rng_state": fit["rng_state"]}
        case = record["case"]
        if case not in exact:
            exact[case] = equality
        if equality != exact[case]:
            raise RuntimeError(f"Exact model/prediction/quality/data/RNG mismatch: {case}, {record['arm']}")
        if external and (case not in external or equality != external[case]["equality"]):
            raise RuntimeError(f"Mismatch with external baseline artifacts: {case}, {record['arm']}")
        if case not in exported and record["arm"] == args.baseline_arm:
            exported[case] = dict(directory=str(directory), prefix=prefix, equality=equality,
                                  identity=ident, params=params)
        wanted = arm.get("expected_gpu_tree")
        if isinstance(wanted, dict):
            wanted = wanted.get(case)
        if wanted is not None and fit["profile"]["tuning"]["gpu_tree"] != wanted:
            raise RuntimeError("Actual GPU driver differs from explicit arm expectation")
        fits.append(fit)
    if statistics.median(f["train_seconds"] for f in fits) != result["median_train_seconds"]:
        raise RuntimeError("Worker median mismatch")
    return {**record, "fits": fits, "params": params, "identity": ident,
            "result_sha256": sha(directory / "result.json"), "exact_artifacts_verified": True}


def paired(ratios):
    ratios = np.asarray(ratios, dtype=np.float64)
    value = dict(cycles=len(ratios), paired_cycle_ratios=ratios.tolist(),
                 geometric_mean_speedup=float(np.exp(np.log(ratios).mean())), wins=int((ratios > 1).sum()))
    if len(ratios) >= 2:
        indices = np.random.default_rng(42).integers(0, len(ratios), (10000, len(ratios)))
        draws = np.exp(np.log(ratios)[indices].mean(axis=1))
        value.update(bootstrap_95_interval=np.percentile(draws, [2.5, 97.5]).tolist(),
                     bootstrap_draws=10000, bootstrap_seed=42, bootstrap_unit="whole paired cycles")
    return value


def summarize(audited, arms, baseline_arm):
    result = {}
    for case in dict.fromkeys(r["case"] for r in audited):
        case_rows = [r for r in audited if r["case"] == case]
        item = dict(arms={}, paired_median_comparisons=[], secondary_mean_comparisons=[])
        for arm in arms:
            rows = [r for r in case_rows if r["arm"] == arm]
            fits = [f for r in rows for f in r["fits"]]
            times = [f["train_seconds"] for f in fits]
            gpu_keys = sorted(set().union(*(f["profile"].get("gpu_seconds", {}) for f in fits)))
            tuning_keys = sorted(set().union(*(f["profile"]["tuning"] for f in fits)))
            item["arms"][arm] = dict(process_count=len(rows), fit_count=len(times),
                individual_fit_seconds=times, individual_fit_mean_seconds=statistics.mean(times),
                individual_fit_median_seconds=statistics.median(times),
                individual_fit_min_seconds=min(times), individual_fit_max_seconds=max(times),
                individual_fit_p95_seconds=float(np.percentile(times, 95)),
                process_medians_seconds={str(r["cycle"]): statistics.median(f["train_seconds"] for f in r["fits"]) for r in rows},
                process_means_seconds={str(r["cycle"]): statistics.mean(f["train_seconds"] for f in r["fits"]) for r in rows},
                rmse_values=sorted(set(f["rmse"] for f in fits)),
                tuning_values={key: [json.loads(s) for s in sorted(set(json.dumps(f["profile"]["tuning"].get(key), sort_keys=True) for f in fits))]
                               for key in tuning_keys},
                gpu_seconds_medians={key: statistics.median(f["profile"]["gpu_seconds"][key] for f in fits if key in f["profile"].get("gpu_seconds", {})) for key in gpu_keys},
                gpu_seconds_means={key: statistics.mean(f["profile"]["gpu_seconds"][key] for f in fits if key in f["profile"].get("gpu_seconds", {})) for key in gpu_keys})
        for arm in (name for name in arms if name != baseline_arm):
            base, candidate = item["arms"][baseline_arm], item["arms"][arm]
            for field, destination in (("process_medians_seconds", "paired_median_comparisons"),
                                       ("process_means_seconds", "secondary_mean_comparisons")):
                if base[field].keys() != candidate[field].keys():
                    raise RuntimeError("Unpaired cycles")
                comparison = paired([base[field][c] / candidate[field][c] for c in sorted(base[field], key=int)])
                comparison.update(baseline_arm=baseline_arm, candidate_arm=arm,
                                  analysis="primary process median" if "medians" in field else "secondary sensitivity; every fit retained")
                if destination == "secondary_mean_comparisons":
                    comparison.update(pooled_fit_mean_speedup=base["individual_fit_mean_seconds"] / candidate["individual_fit_mean_seconds"],
                                      pooled_fit_p95_speedup=base["individual_fit_p95_seconds"] / candidate["individual_fit_p95_seconds"])
                item[destination].append(comparison)
        result[case] = item
    return dict(cases=result, process_count=len(audited), fit_count=sum(len(r["fits"]) for r in audited),
                all_models_predictions_quality_data_rng_exact_within_case=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-file", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", help="Select named arms in initial order; default all file entries")
    parser.add_argument("--baseline-arm", help="Baseline for comparisons/export; default first selected arm")
    parser.add_argument("--common-params", default="{}", help="JSON controls applied before each arm's params")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(DEFAULT_CASES))
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--compare-baseline", type=Path,
                        help="Earlier baseline-artifacts.json; saved files and hashes are reverified")
    parser.add_argument("--plan-only", action="store_true", help="Validate inputs/pins and print plan; no output/GPU")
    args = parser.parse_args()
    if min(args.cycles, args.repeats, args.rows, args.bins, args.rounds, args.warmup_rounds) < 1 or args.features < 4:
        parser.error("Need positive sizes/counts and at least four features")
    if sys.flags.optimize or len(args.cases) != len(set(args.cases)):
        parser.error("Disable Python optimization and select distinct cases")
    common = json.loads(args.common_params)
    arms_file = args.arms_file.expanduser().resolve()
    defined = json.loads(arms_file.read_text())
    if not isinstance(common, dict) or not isinstance(defined, dict) or not defined:
        parser.error("Arm file and common params must be JSON objects")
    selected = args.arms or list(defined)
    if len(set(selected)) != len(selected) or any(name not in defined for name in selected):
        parser.error("Select distinct, known arm names")
    args.baseline_arm = args.baseline_arm or selected[0]
    if args.baseline_arm not in selected:
        parser.error("Baseline arm must be selected")
    arms = {}

    def arm_path(value):
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else arms_file.parent / path).resolve())

    for name in selected:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            parser.error("Arm names must be filesystem-safe letters/digits/underscores/hyphens")
        arm = dict(defined[name])
        for key in ("package_root", "native_path"):
            arm[key] = arm_path(arm[key])
        if arm.get("source_root"):
            arm["source_root"] = arm_path(arm["source_root"])
        pin = arm["native_sha256"]
        if len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            parser.error("Every arm needs a lowercase SHA256 pin")
        arm["params"] = {**common, **arm.get("params", {})}
        expected = arm.get("expected_gpu_tree")
        # JSON booleans are ints in Python; the worker accepts only 0 or 1.
        if expected is not None and not (expected in (0, 1) if type(expected) is int else
                isinstance(expected, dict) and all(k in CASES and type(v) is int and v in (0, 1)
                                                   for k, v in expected.items())):
            parser.error("expected_gpu_tree must be 0/1 or case-to-0/1 mapping")
        arms[name] = arm
    worker = HERE / "deep_tree_worker.py"
    worker_sha = sha(worker)
    quick_pins(arms, worker, worker_sha)
    external = {}
    if args.compare_baseline:
        baseline_record = json.loads(args.compare_baseline.read_text())
        if baseline_record.get("status") != "passed":
            raise RuntimeError("External baseline must come from a completed, passed run")
        external = baseline_record["cases"]
        for case in args.cases:
            row = external[case]
            actual = outputs(Path(row["directory"]), row["prefix"])
            for field, value in actual.items():
                if value != row["equality"][field]:
                    raise RuntimeError(f"External baseline artifact changed: {case}, {field}")
    output = args.output.resolve()
    plan = []
    for cycle in range(1, args.cycles + 1):
        rotation = (cycle - 1) % len(args.cases)
        cases = args.cases[rotation:] + args.cases[:rotation]
        order = selected if cycle % 2 else list(reversed(selected))
        for case in cases:
            for name in order:
                arm, definition = arms[name], CASES[case]
                stem = f"{case}-cycle{cycle:03d}-{name}"
                directory = output / "processes" / stem
                command = [sys.executable, "-B", str(worker), "--arm", name, "--case", case,
                    "--package-root", arm["package_root"], "--native-path", arm["native_path"],
                    "--native-sha256", arm["native_sha256"], "--output", str(directory),
                    "--depth", str(definition["depth"]), "--density", definition["density"],
                    "--driver", definition["driver"], "--params-json", json.dumps(arm["params"], sort_keys=True),
                    "--rows", str(args.rows), "--features", str(args.features), "--bins", str(args.bins),
                    "--rounds", str(args.rounds), "--warmup-rounds", str(args.warmup_rounds),
                    "--repeats", str(args.repeats)]
                expected = arm.get("expected_gpu_tree")
                if isinstance(expected, dict):
                    expected = expected.get(case)
                if expected is not None:
                    command += ["--expected-gpu-tree", str(expected)]
                plan.append(dict(case=case, cycle=cycle, arm=name, stem=stem,
                                 directory=str(directory), command=command))
    if args.plan_only:
        print(json.dumps(dict(arms=arms, plan=plan, planned_processes=len(plan),
                              planned_fits=len(plan) * args.repeats), indent=2))
        return
    snapshots = {name: arm_snapshot(arm) for name, arm in arms.items()}
    output.mkdir(parents=True, exist_ok=False)
    (output / "processes").mkdir()
    (output / "receipts").mkdir()
    (output / "deep_tree_worker-as-run.py").write_bytes(worker.read_bytes())
    (output / "deep_trees-as-run.py").write_bytes(Path(__file__).read_bytes())
    save(output / "arms-resolved.json", arms)
    environment = {key: value for key, value in os.environ.items()
                   if key not in REMOVE and not key.startswith(("MTL_", "METALGB_", "XGB_METAL_"))}
    environment.update(OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    provenance = dict(started_utc=now(), python=sys.executable, python_version=platform.python_version(),
        platform=platform.platform(), arms=arms, arm_snapshots=snapshots, worker_sha256=worker_sha,
        orchestrator_sha256=sha(__file__), plan=plan, planned_processes=len(plan),
        planned_fits=len(plan) * args.repeats,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        external_baseline=None if not args.compare_baseline else
            dict(path=str(args.compare_baseline.resolve()), sha256=sha(args.compare_baseline)))
    for key, command in (("cpu", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                         ("power", ["pmset", "-g", "custom"])):
        probe = subprocess.run(command, capture_output=True, text=True)
        provenance[key] = dict(command=command, exit_code=probe.returncode, stdout=probe.stdout, stderr=probe.stderr)
    save(output / "provenance-before.json", provenance)
    audited, exact, exported = [], {}, {}
    summary = dict(status="running", started_utc=now(), planned_processes=len(plan), planned_fits=len(plan) * args.repeats)
    try:
        for sequence, record in enumerate(plan, 1):
            quick_pins(arms, worker, worker_sha)
            arm = arms[record["arm"]]
            child_env = {**environment, "PYTHONPATH": arm["package_root"]}
            stem = record["stem"]
            log = output / "receipts" / (stem + ".log")
            receipt = {**record, "sequence": sequence, "started_utc": now(), "log": str(log)}
            print(f"[{sequence}/{len(plan)}] {stem}", flush=True)
            began = time.perf_counter()
            with log.open("x") as stream:
                process = subprocess.Popen(record["command"], cwd=HERE, env=child_env,
                                           stdout=stream, stderr=subprocess.STDOUT)
                receipt["pid"] = process.pid
                save(output / "receipts" / (stem + ".started.json"), receipt)
                code = process.wait()
            receipt.update(exit_code=code, finished_utc=now(), elapsed_seconds=time.perf_counter() - began,
                           log_sha256=sha(log))
            save(output / "receipts" / (stem + ".exit.json"), receipt)
            if code:
                raise RuntimeError(f"Worker failed ({code}): {log}")
            quick_pins(arms, worker, worker_sha)
            current = audit(receipt, arm, worker_sha, args, exact, exported, external)
            audited.append(current)
            save(output / "audit-progress.json", dict(completed_processes=len(audited), processes=audited))
            print("  median %.6fs; exact outputs verified" % statistics.median(f["train_seconds"] for f in current["fits"]), flush=True)
        if len(audited) != len(plan):
            raise RuntimeError("Missing scheduled processes")
        summary.update(summarize(audited, selected, args.baseline_arm), status="passed")
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}",
                       traceback=traceback.format_exc(), completed_processes=len(audited))
        raise
    finally:
        after = {name: arm_snapshot(arm) for name, arm in arms.items()}
        unchanged = after == snapshots and sha(worker) == worker_sha
        if not unchanged:
            summary.update(status="failed", provenance_error="Arm package/native/source or worker changed")
        summary.update(finished_utc=now(), source_packages_natives_unchanged=unchanged,
                       external_baseline_compared=bool(external))
        save(output / "provenance-after.json", dict(arm_snapshots=after, unchanged=unchanged, completed_utc=now()))
        save(output / "summary.json", summary)
        save(output / "artifact-audit.json", dict(processes=audited, case_exact_outputs=exact))
        save(output / "baseline-artifacts.json", dict(baseline_arm=args.baseline_arm,
             status=summary["status"], cases=exported, note="Absolute artifact paths; preserve the referenced process directories"))
        save(output / "manifest.json", dict(completed_utc=now(), files_sha256={str(p.relative_to(output)): sha(p)
             for p in sorted(output.rglob("*")) if p.is_file()}))
    if summary["status"] != "passed":
        raise RuntimeError("Final provenance check failed")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
