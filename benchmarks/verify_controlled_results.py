#!/usr/bin/env python3
"""Audit the committed November 2026 benchmark receipts and print their medians."""
from collections import defaultdict
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parent / "results/2026-11-12"
CASES = ("dense100", "dense262", "missing100")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    archives = {name: json.loads((ROOT / filename).read_text()) for name, filename in {
        "comparison": "controlled-comparison.json", "cpu": "cpu-scaling.json",
        "wait": "cpu-wait-policy.json", "gpu": "gpu-tuning.json"}.items()}
    for name, expected in (("comparison", 54), ("cpu", 162), ("wait", 18), ("gpu", 315)):
        require(len(archives[name]["runs"]) == expected, f"{name}: incorrect receipt count")
    hardware = archives["comparison"]["machines"]
    for field in ("stock_cpu_wheel_sha256", "openmp_sha256", "source_commit", "numpy"):
        require(len({hardware[m][field] for m in ("m5", "m4", "m1")}) == 1,
                f"Machine manifests disagree on {field}")
    all_records = [row["receipt"] for archive in archives.values() for row in archive["runs"]]
    for case in CASES:
        selected = [r for r in all_records if r["case"] == case]
        require(bool(selected), f"Missing case {case}")
        for field in ("input_sha256", "label_sha256", "prediction_sha256"):
            require(len({r[field] for r in selected}) == 1, f"{case}: {field} mismatch")
        gpu = [row["receipt"] for row in archives["gpu"]["runs"] if row["receipt"]["case"] == case]
        require(len({r["model_sha256"] for r in gpu}) == 1, f"{case}: Metal model mismatch")
    print("Controlled comparison: median training milliseconds; CPU / Metal")
    groups = defaultdict(list)
    for row in archives["comparison"]["runs"]:
        record = row["receipt"]
        groups[(record["case"], row["machine"], record["backend"])].append(record["train_seconds"])
    for case in CASES:
        for machine in ("m5", "m4", "m1"):
            values = {}
            for backend in ("metal", "cpu"):
                samples = groups[(case, machine, backend)]
                require(len(samples) == 3, f"{case}/{machine}/{backend}: expected three fits")
                values[backend] = statistics.median(samples) * 1000
            print(f"{case:10} {machine:2} GPU {values['metal']:8.1f} CPU {values['cpu']:8.1f} "
                  f"ratio {values['cpu'] / values['metal']:.2f}x")
    print("M5 CPU thread sweep: fastest measured median per workload")
    groups = defaultdict(list)
    for row in archives["cpu"]["runs"]:
        if row["machine"] == "m5":
            record = row["receipt"]
            groups[(record["case"], record["threads"])].append(record["train_seconds"])
    for case in CASES:
        candidates = {}
        for threads in (1, 4, 8, 12, 16, 20, 24, 28, 32, 36):
            samples = groups[(case, threads)]
            require(len(samples) == 3, f"{case}/{threads}: expected three CPU sweep fits")
            candidates[threads] = statistics.median(samples)
        best = min(candidates, key=candidates.get)
        print(f"{case:10} {best:2} threads {candidates[best] * 1000:8.1f} ms")
    for phase, expected in (("threads", 180), ("hist", 135)):
        phase_runs = [r for r in archives["gpu"]["runs"] if r["phase"] == phase]
        require(len(phase_runs) == expected, f"{phase}: incorrect GPU receipt count")
        for row in phase_runs:
            record = row["receipt"]
            require(record["tuning"] == record["profile"].get("tuning"),
                    f"{row['name']}: GPU profile mismatch")
            require(record["tuning"].get("gpu_tree") == 1, f"{row['name']}: GPU tree not selected")
    print("Verified 549 receipts, machine pins, and input/label/prediction hashes.")


if __name__ == "__main__":
    main()
