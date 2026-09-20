"""Repeatable end-to-end training and inference measurements on the actual device."""
from pathlib import Path
import argparse
import ctypes
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
import numpy as np
import xgboost as xgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--targets", type=int, default=1, help="Number of square-error regression outputs")
    parser.add_argument("--multi-strategy", choices=("one_output_per_tree", "multi_output_tree"), default="one_output_per_tree")
    parser.add_argument("--feature-levels", type=int, help="Use this many discrete feature values to compare identical quantile cuts")
    parser.add_argument("--weighted", action="store_true", help="Use reproducible fractional row weights for square-error regression")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--booster", choices=("gbtree", "gblinear"), default="gbtree")
    parser.add_argument("--linear-updater", choices=("shotgun", "coord_descent"), default="coord_descent")
    parser.add_argument("--feature-selector", choices=("cyclic", "shuffle", "random", "greedy", "thrifty"), default="cyclic")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--response", choices=("quadratic", "linear"), default="quadratic")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--grow-policy", choices=("depthwise", "lossguide"), default="depthwise")
    parser.add_argument("--max-leaves", type=int, default=0)
    parser.add_argument("--max-cached-hist-node", type=int, help="Bound the GPU histogram cache; the automatic GPU default is 4096 nodes")
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--subsample", type=float, help="Fraction of training rows to sample")
    parser.add_argument("--sampling-method", choices=("uniform", "gradient_based"), help="Row sampling method")
    parser.add_argument("--num-parallel-tree", type=int, default=1)
    parser.add_argument("--max-delta-step", type=float, help="Bound leaf updates; use 1 for the Cox benchmark's nearly deterministic event times")
    parser.add_argument("--objective", choices=("reg:squarederror", "reg:absoluteerror", "reg:quantileerror", "reg:expectileerror", "reg:normal", "survival:aft", "survival:cox", "rank:ndcg", "rank:map", "rank:pairwise"),
                        default="reg:squarederror")
    parser.add_argument("--quantile-alpha", type=float, nargs="+", help="Sorted quantiles, default: 0.1 0.5 0.9")
    parser.add_argument("--expectile-alpha", type=float, nargs="+", help="Sorted expectiles, default: 0.1 0.5 0.9")
    parser.add_argument("--aft-distribution", choices=("normal", "logistic", "extreme"), help="AFT noise distribution, default: normal")
    parser.add_argument("--aft-scale", type=float, help="Positive AFT scale for this benchmark, default: 1")
    parser.add_argument("--query-size", type=int, help="Documents per ranking query, default: 128")
    parser.add_argument("--ranking-pair-method", choices=("topk", "mean"), help="Ranking pair method, default: topk")
    parser.add_argument("--ranking-pairs", type=int, help="Pairs per document, default: 32 for topk or 1 for mean")
    parser.add_argument("--ranking-unbiased", action="store_true", help="Enable ranking position-bias updates")
    parser.add_argument("--no-ranking-cache", action="store_true", help="Allocate ranking workspaces on each iteration")
    parser.add_argument("--no-ranking-aggregation", action="store_true", help="Issue one atomic update per ranking pair endpoint")
    parser.add_argument("--rate-drop", type=float, help="Fraction of trees to drop during training")
    parser.add_argument("--skip-drop", type=float, help="Probability of skipping tree dropout")
    parser.add_argument("--one-drop", action="store_true", help="Drop at least one tree when dropout is not skipped")
    parser.add_argument("--sample-type", choices=("uniform", "weighted"), help="Tree dropout sampling method")
    parser.add_argument("--normalize-type", choices=("tree", "forest"), help="Tree dropout weight normalization")
    parser.add_argument("--histogram-rows", type=int)
    parser.add_argument("--features-per-group", type=int)
    parser.add_argument("--no-histogram-subtraction", action="store_true")
    parser.add_argument("--no-partition-aggregation", action="store_true")
    parser.add_argument("--atomic-partition", action="store_true")
    parser.add_argument("--separate-partition", action="store_true", help="Keep CPU planning and a separate partition command")
    parser.add_argument("--coalesce-encoders", action="store_true", help="Reuse a serial compute encoder within each tree command")
    parser.add_argument("--compute-fill", action="store_true", help="Use a compute kernel to clear tree buffers")
    parser.add_argument("--no-tree-workspace", action="store_true", help="Allocate temporary tree buffers separately for each growth batch")
    parser.add_argument("--no-tree-persistence", action="store_true", help="Release temporary tree buffers between trees")
    parser.add_argument("--serial-splits", action="store_true", help="Use the serial feature scan for split search")
    parser.add_argument("--serial-cuts", action="store_true", help="Use ordered serial GPU cut selection for comparison")
    parser.add_argument("--serial-summary", action="store_true", help="Use ordered serial GPU batch summarization for comparison")
    parser.add_argument("--force-parallel-summary", action="store_true", help="Test parallel summaries from 32 eligible runs; the default requires a prunable batch with at least 32768 runs")
    parser.add_argument("--scalar-split-bin-tile", type=int, help="Numeric scalar bins per SIMD group; zero selects the untiled comparison path")
    parser.add_argument("--vector-split-bin-tile", type=int, help="Vector candidate bins per SIMD group; zero selects the untiled comparison path")
    parser.add_argument("--build-record", type=Path, help="Use verified provenance from a prior report when measuring an archived wheel")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-result", action="store_true", help="Save complete predictions, the last model and tree feature cuts next to --output")
    args = parser.parse_args()
    if min(args.rows, args.features, args.targets, args.rounds, args.repeats) < 1:
        parser.error("Dataset sizes, rounds, and repeats must be positive.")
    if args.targets != 1 and args.objective != "reg:squarederror":
        parser.error("--targets requires --objective reg:squarederror.")
    if args.weighted and args.objective != "reg:squarederror":
        parser.error("--weighted requires --objective reg:squarederror.")
    if args.multi_strategy == "multi_output_tree" and args.booster != "gbtree":
        parser.error("Vector-leaf benchmarking requires --booster gbtree.")
    if args.scalar_split_bin_tile is not None:
        tile = args.scalar_split_bin_tile
        if tile and (tile < 32 or tile > 16384 or tile % 32):
            parser.error("--scalar-split-bin-tile must be zero or a multiple of 32 in [32, 16384].")
        if args.multi_strategy != "one_output_per_tree" or args.booster != "gbtree":
            parser.error("--scalar-split-bin-tile requires scalar tree leaves.")
    if args.vector_split_bin_tile is not None:
        tile = args.vector_split_bin_tile
        if tile and (tile < 32 or tile > 16384 or tile % 32):
            parser.error("--vector-split-bin-tile must be zero or a multiple of 32 in [32, 16384].")
        if args.multi_strategy != "multi_output_tree":
            parser.error("--vector-split-bin-tile requires --multi-strategy multi_output_tree.")
    if args.feature_levels is not None and args.feature_levels < 2:
        parser.error("--feature-levels must be at least two.")
    if args.subsample is not None and (not np.isfinite(args.subsample) or not 0 <= args.subsample <= 1):
        parser.error("--subsample must be finite and in [0, 1].")
    if args.num_parallel_tree < 1:
        parser.error("--num-parallel-tree must be positive.")
    if args.booster == "gblinear" and (args.subsample is not None or args.sampling_method is not None or args.num_parallel_tree != 1):
        parser.error("Row sampling and parallel trees require --booster gbtree.")
    if args.save_result and args.output is None:
        parser.error("--save-result requires --output.")
    if args.booster == "gbtree" and (args.depth < 0 or args.max_leaves < 0 or not (args.depth or args.max_leaves)):
        parser.error("Depth and leaf limits must be nonnegative, with at least one constrained.")
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative.")
    if args.booster == "gblinear" and args.linear_updater == "shotgun" and args.feature_selector not in ("cyclic", "shuffle"):
        parser.error("The shotgun updater supports cyclic and shuffle feature selection.")
    if args.max_cached_hist_node is not None and args.max_cached_hist_node < 1:
        parser.error("--max-cached-hist-node must be positive.")
    if args.max_delta_step is not None and (not np.isfinite(args.max_delta_step) or args.max_delta_step < 0):
        parser.error("--max-delta-step must be finite and nonnegative.")
    for value in (args.rate_drop, args.skip_drop):
        if value is not None and (not np.isfinite(value) or value < 0 or value > 1):
            parser.error("Dropout probabilities must be finite and in [0, 1].")
    if args.quantile_alpha is not None and args.objective != "reg:quantileerror":
        parser.error("--quantile-alpha requires --objective reg:quantileerror.")
    if args.expectile_alpha is not None and args.objective != "reg:expectileerror":
        parser.error("--expectile-alpha requires --objective reg:expectileerror.")
    if (args.aft_distribution is not None or args.aft_scale is not None) and args.objective != "survival:aft":
        parser.error("AFT distribution and scale options require --objective survival:aft.")
    if args.aft_scale is not None and (not np.isfinite(args.aft_scale) or args.aft_scale <= 0):
        parser.error("The survival benchmark requires a positive finite scale.")
    if not args.objective.startswith("rank:") and any((args.query_size is not None, args.ranking_pair_method is not None,
                                                       args.ranking_pairs is not None, args.ranking_unbiased,
                                                       args.no_ranking_cache, args.no_ranking_aggregation)):
        parser.error("Ranking options require a rank: objective.")
    if (args.query_size is not None and args.query_size < 1) or (args.ranking_pairs is not None and args.ranking_pairs < 1):
        parser.error("Query size and ranking pair count must be positive.")
    requested_levels = args.expectile_alpha if args.objective == "reg:expectileerror" else args.quantile_alpha
    levels = np.asarray(requested_levels if requested_levels is not None else [0.1, 0.5, 0.9], dtype=np.float32)
    if not np.isfinite(levels).all() or np.any((levels < 0) | (levels > 1)) or np.any(np.diff(levels) < 0):
        parser.error("Alpha levels must be finite, sorted, and in [0, 1].")
    output_groups = 2 if args.objective == "reg:normal" else len(levels) if args.objective in ("reg:quantileerror", "reg:expectileerror") else args.targets
    rng = np.random.default_rng(43)
    if args.feature_levels is None:
        X = rng.normal(size=(args.rows, args.features)).astype(np.float32)
        feature_generation = "standard normal, NumPy default_rng(43)"
    else:
        X = rng.integers(0, args.feature_levels, size=(args.rows, args.features)).astype(np.float32)
        X = (X - np.float32((args.feature_levels - 1) / 2)) / np.float32((args.feature_levels - 1) / 4)
        feature_generation = f"{args.feature_levels} equally spaced float32 levels in [-2, 2], NumPy default_rng(43)"
    noise = rng.normal(0, 0.1, args.rows)
    y = (2 * X[:, 0] + X[:, min(1, args.features - 1)] ** 2 + noise).astype(np.float32)
    metadata = {"label": y}
    response_generation = "2 * X[:, 0] + X[:, min(1, features - 1)] ** 2 + normal(0, 0.1)"
    if args.response == "linear":
        y = (2 * X[:, 0] - 0.75 * X[:, min(1, args.features - 1)] + noise).astype(np.float32)
        metadata = {"label": y}
        response_generation = "2 * X[:, 0] - 0.75 * X[:, min(1, features - 1)] + normal(0, 0.1)"
    if args.targets > 1:
        outputs = [y]
        for target in range(1, args.targets):
            first = X[:, target % args.features]
            second = X[:, (target + 1) % args.features]
            response = 2 * first - 0.75 * second if args.response == "linear" else 2 * first + second ** 2
            outputs.append((response + rng.normal(0, 0.1, args.rows)).astype(np.float32))
        y = np.column_stack(outputs)
        metadata = {"label": y}
        response_generation += "; additional outputs shift feature indices by target modulo feature count, with independent normal(0, 0.1) noise"
    params = {"max_depth": args.depth, "max_bin": args.bins, "seed": 9,
              "tree_method": "hist", "grow_policy": args.grow_policy, "max_leaves": args.max_leaves}
    if args.booster == "gblinear":
        params = {"booster": "gblinear", "updater": args.linear_updater, "feature_selector": args.feature_selector,
                  "seed": 9, "top_k": args.top_k}
    elif args.multi_strategy != "one_output_per_tree":
        params["multi_strategy"] = args.multi_strategy
    if args.subsample is not None:
        params["subsample"] = args.subsample
    if args.sampling_method is not None:
        params["sampling_method"] = args.sampling_method
    if args.num_parallel_tree != 1:
        params["num_parallel_tree"] = args.num_parallel_tree
    if args.max_cached_hist_node is not None:
        params["max_cached_hist_node"] = args.max_cached_hist_node
    for name in ("rate_drop", "skip_drop", "sample_type", "normalize_type"):
        value = getattr(args, name)
        if value is not None:
            params[name] = value
    if args.one_drop:
        params["one_drop"] = True
    if args.max_delta_step is not None:
        params['max_delta_step'] = args.max_delta_step
    if args.objective != "reg:squarederror":
        params["objective"] = args.objective
    if args.objective.startswith("rank:"):
        query_size = min(args.query_size or 128, len(y))
        method = args.ranking_pair_method or "topk"
        pairs = args.ranking_pairs if args.ranking_pairs is not None else (32 if method == "topk" else 1)
        params.update(lambdarank_pair_method=method, lambdarank_num_pair_per_sample=pairs,
                      lambdarank_unbiased=int(args.ranking_unbiased))
        label = (y > 0).astype(np.float32) if args.objective == "rank:map" else np.digitize(y, [-2, -0.5, 0.5, 2]).astype(np.float32)
        metadata = {"label": label, "qid": np.arange(len(y), dtype=np.uint32) // query_size}
        response_generation += f"; relevance = {'response > 0' if args.objective == 'rank:map' else 'digitize(response, [-2, -0.5, 0.5, 2])'}; consecutive queries of {query_size} documents"
        cutoff = pairs if method == "topk" else (32 if args.objective == "rank:map" else query_size)
        n_queries = (len(y) + query_size - 1) // query_size
        padded_labels = np.pad(label.astype(np.float64), (0, n_queries * query_size - len(y))).reshape(n_queries, query_size)
        n = min(cutoff, query_size)
        discount = 1 / np.log2(np.arange(n, dtype=np.float64) + 2)
        ideal = np.sort(padded_labels, axis=1)[:, ::-1][:, :n]
        idcg = np.sum(np.expm1(ideal * np.log(2)) * discount, axis=1)
        loss_name = ("map" if args.objective == "rank:map" else "ndcg") + f"@{cutoff}-loss"
        baseline = np.float32(0)
        def loss(prediction):
            values = np.broadcast_to(np.asarray(prediction, dtype=np.float64), (len(y),))
            padded = np.pad(values, (0, n_queries * query_size - len(y)), constant_values=-np.inf).reshape(n_queries, query_size)
            order = np.argsort(-padded, axis=1, kind="stable")[:, :n]
            ranked = np.take_along_axis(padded_labels, order, axis=1)
            if args.objective == "rank:map":
                hits = np.cumsum(ranked, axis=1, dtype=np.float64)
                score = np.sum(hits / (np.arange(n) + 1) * ranked, axis=1)
                total = np.minimum(padded_labels.sum(axis=1), cutoff)
            else:
                score = np.sum(np.expm1(ranked * np.log(2)) * discount, axis=1)
                total = idcg
            score = np.divide(score, total, out=np.ones_like(score), where=total > 0)
            return float(1 - np.mean(score, dtype=np.float64))
    elif args.objective == "survival:aft":
        from scipy.special import erf
        distribution = args.aft_distribution or "normal"
        scale = args.aft_scale if args.aft_scale is not None else 1.0
        params.update(aft_loss_distribution=distribution, aft_loss_distribution_scale=scale)
        times = np.exp(np.float32(0.3) * y).astype(np.float32)
        lower, upper = times.copy(), times.copy()
        upper[1::4] = np.inf
        lower[2::4] = 0
        lower[3::4] *= np.float32(0.8)
        upper[3::4] *= np.float32(1.2)
        metadata = {"label_lower_bound": lower, "label_upper_bound": upper}
        response_generation += "; times = exp(0.3 * response); row modulo 4 selects exact, right, left, or [0.8t, 1.2t] interval censoring"
        loss_name, baseline = "aft-nloglik", np.float32(0.5)
        def loss(prediction):
            margin = np.log(np.broadcast_to(np.asarray(prediction, dtype=np.float64), (len(y),)))
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                zl = (np.log(lower.astype(np.float64)) - margin) / scale
                zu = (np.log(upper.astype(np.float64)) - margin) / scale
                def pdf(z):
                    if distribution == "normal":
                        return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
                    w = np.exp(z)
                    if distribution == "logistic":
                        return np.where(np.isinf(w * w), 0, w / ((1 + w) * (1 + w)))
                    return np.where(np.isinf(w), 0, w * np.exp(-w))
                def cdf(z):
                    if distribution == "normal":
                        return 0.5 * (1 + erf(z / np.sqrt(2)))
                    w = np.exp(z)
                    if distribution == "logistic":
                        return np.where(np.isinf(w), 1, w / (1 + w))
                    return 1 - np.exp(-w)
                probability = np.where(np.isinf(upper), 1, cdf(zu)) - np.where(lower <= 0, 0, cdf(zl))
                probability = np.where(lower == upper, pdf(zl) / (scale * lower.astype(np.float64)), probability)
                return float(np.mean(-np.log(np.fmax(probability, 1e-12)), dtype=np.float64))
    elif args.objective == "survival:cox":
        times = np.exp(np.float32(0.3) * y).astype(np.float32)
        events = rng.uniform(size=len(y)) >= 0.3
        metadata = {"label": np.where(events, times, -times)}
        response_generation += "; times = exp(0.3 * response); independent 30% right censoring uses negative labels"
        order = np.argsort(times, kind="stable")
        sorted_times = times[order]
        starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(sorted_times) != 0], np.arange(len(y)), 0))
        sorted_events = events[order]
        loss_name, baseline = "cox-nloglik", np.float32(1)
        def loss(prediction):
            values = np.broadcast_to(np.asarray(prediction, dtype=np.float64), (len(y),))[order]
            prefix = np.cumsum(values, dtype=np.float64)
            risk = prefix[-1] - np.where(starts == 0, 0, prefix[np.maximum(starts - 1, 0)])
            return float(np.mean((np.log(risk) - np.log(values))[sorted_events], dtype=np.float64))
    elif args.objective == "reg:quantileerror":
        params["quantile_alpha"] = levels.tolist()
        loss_name = "quantile"
        baseline = np.quantile(y, levels, method="inverted_cdf")
        def loss(prediction):
            error = np.asarray(prediction).reshape(-1, len(levels)) - y[:, None]
            return float(np.mean(np.where(error >= 0, (1 - levels) * error, -levels * error), dtype=np.float64))
    elif args.objective == "reg:expectileerror":
        params["expectile_alpha"] = levels.tolist()
        loss_name = "expectile"
        mean = np.float32(np.mean(y, dtype=np.float64))
        difference = mean - y
        baseline = []
        for alpha in levels:
            weight = np.where(difference >= 0, 1 - alpha, alpha)
            step = np.float32(-np.sum(weight * difference, dtype=np.float64) / max(np.sum(weight, dtype=np.float64), 1e-6))
            baseline.append(np.float32(mean + step))
        baseline = np.maximum.accumulate(np.asarray(baseline, dtype=np.float32))
        margins = baseline.copy()
        for j in range(len(levels) - 1, 0, -1):
            gap = max(margins[j] - margins[j - 1] - np.float32(1e-6), np.float32(1e-6))
            margins[j] = gap + np.log(-np.expm1(-gap))
        for j in range(1, len(levels)):
            value = margins[j]
            increment = value + np.log1p(np.exp(-value)) if value > 0 else np.log1p(np.exp(value))
            baseline[j] = baseline[j - 1] + (np.float32(1e-6) + increment)
        def loss(prediction):
            error = np.asarray(prediction).reshape(-1, len(levels)) - y[:, None]
            return float(np.mean((np.where(error >= 0, 1 - levels, levels) * error) * error, dtype=np.float64))
    elif args.objective == "reg:normal":
        loss_name = "normal-nloglik"
        mean = np.float32(np.mean(y, dtype=np.float64))
        variance = np.mean((y.astype(np.float64) - mean) ** 2)
        baseline = np.asarray([mean, np.log(variance + np.finfo(np.float32).eps)], dtype=np.float32)
        def loss(prediction):
            values = np.broadcast_to(np.asarray(prediction).reshape(-1, 2), (len(y), 2))
            residual = y - values[:, 0]
            standardized = np.zeros_like(residual)
            nonzero = residual != 0
            standardized[nonzero] = np.exp(2 * np.log(np.abs(residual[nonzero])) - values[nonzero, 1])
            return float(np.mean(np.float32(0.5) * (np.float32(1.8378770664093453) + values[:, 1] + standardized), dtype=np.float64))
    elif args.objective == "reg:absoluteerror":
        loss_name, baseline = "mae", np.median(y)
        def loss(prediction):
            return float(np.mean(np.abs(prediction - y), dtype=np.float64))
    else:
        loss_name, baseline = "rmse", np.mean(y, axis=0)
        def loss(prediction):
            return float(np.sqrt(np.mean((prediction - y) ** 2)))
    weight_generation = "Implicit unit weights"
    if args.weighted:
        row_weights = np.random.default_rng(144).uniform(0.1, 2, len(y)).astype(np.float32)
        metadata['weight'] = row_weights
        weight_generation = "float32 uniform [0.1, 2), independent NumPy default_rng(144)"
        weights64 = row_weights.astype(np.float64)
        labels64 = np.asarray(y, np.float64).reshape(len(y), -1)
        baseline = np.average(labels64, axis=0, weights=weights64)
        def loss(prediction):
            values = np.asarray(prediction, np.float64).reshape(-1, labels64.shape[1])
            per_row = np.mean((values - labels64) ** 2, axis=1)
            return float(np.sqrt(np.average(per_row, weights=weights64)))
    baseline_loss = loss(baseline)
    assert np.isfinite(baseline_loss), "The constant predictor must have a finite loss."
    build = xgb.build_info()
    is_metal = build.get("USE_METAL", False)
    library_sha256 = hashlib.sha256(Path(build['libxgboost']).read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    provenance = {'library_sha256': library_sha256, 'xgboost_version': xgb.__version__}
    if args.build_record:
        prior = json.loads(args.build_record.read_text())
        if prior['library_sha256'] != library_sha256:
            parser.error("The archived build record does not match the loaded library hash.")
        for key in ('native_source_sha256', 'reference_commit'):
            if key in prior:
                provenance[key] = prior[key]
        provenance['build_record'] = str(args.build_record)
    elif is_metal:
        source_hash = hashlib.sha256()
        for path in sorted([*root.glob('src/*'), *root.glob('include/metalgb/*'),
                            *root.glob('kernels/*.metal*'), root / 'CMakeLists.txt']):
            if path.is_file():
                source_hash.update(str(path.relative_to(root)).encode())
                source_hash.update(path.read_bytes())
        provenance['native_source_sha256'] = source_hash.hexdigest()
    else:
        checkout = Path(xgb.__file__).resolve().parents[2]
        if (checkout / '.git').exists():
            provenance['reference_commit'] = subprocess.check_output(
                ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if is_metal:
        if args.objective.startswith("rank:"):
            params["metal_ranking_cache"] = int(not args.no_ranking_cache)
            params["metal_ranking_aggregation"] = int(not args.no_ranking_aggregation)
    if is_metal and args.booster == "gbtree":
        if args.histogram_rows is not None:
            params["metal_histogram_rows"] = args.histogram_rows
        if args.features_per_group is not None:
            params["metal_features_per_group"] = args.features_per_group
        params["metal_histogram_subtraction"] = int(not args.no_histogram_subtraction)
        params["metal_partition_aggregation"] = int(not args.no_partition_aggregation)
        params["metal_partition_scan"] = int(not args.atomic_partition)
        params["metal_fused_partition"] = int(not args.separate_partition)
        params["metal_coalesce_encoders"] = int(args.coalesce_encoders)
        params["metal_compute_fill"] = int(args.compute_fill)
        params["metal_tree_workspace"] = int(not args.no_tree_workspace)
        if args.no_tree_persistence:
            params["metal_tree_persistent_workspace"] = 0
        params["metal_parallel_splits"] = int(not args.serial_splits)
        if args.serial_cuts:
            os.environ["METALGB_SERIAL_CUTS"] = "1"
        if args.serial_summary:
            os.environ["METALGB_SERIAL_SUMMARY"] = "1"
        if args.force_parallel_summary:
            os.environ["METALGB_FORCE_PARALLEL_SUMMARY"] = "1"
        if args.scalar_split_bin_tile is not None:
            params["metal_scalar_split_bin_tile"] = args.scalar_split_bin_tile
        if args.vector_split_bin_tile is not None:
            params["metal_vector_split_bin_tile"] = args.vector_split_bin_tile
    # Compile/cache pipelines before measurement, then use fresh data each repeat.
    warmup = dict(params)
    if is_metal and args.booster == "gbtree" and args.features_per_group is None:
        warmup['metal_features_per_group'] = 4 if args.rows >= 262144 and args.features >= 32 and args.bins <= 256 else 1
    warmup_rows = min(1024, len(X))
    warmup_data = xgb.DMatrix(X[:warmup_rows], **{key: value[:warmup_rows] for key, value in metadata.items()})
    warmup_model = xgb.train(warmup, warmup_data, 1)
    warmup_model.predict(warmup_data)
    del warmup_model, warmup_data
    measurements = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        data = xgb.DMatrix(X, **metadata)
        ingested = time.perf_counter()
        model = xgb.train(params, data, args.rounds, verbose_eval=False)
        trained = time.perf_counter()
        prediction = model.predict(data)
        predicted = time.perf_counter()
        assert np.isfinite(prediction).all()
        value = loss(prediction)
        assert np.isfinite(value), "Training produced a nonfinite loss."
        assert value < baseline_loss or (value == baseline_loss == 0), "Training failed to improve on the constant predictor."
        measurement = {"ingest_seconds": ingested - start, "train_seconds": trained - ingested,
                       "predict_seconds": predicted - trained, loss_name: value,
                       "model_sha256": hashlib.sha256(model.save_raw(raw_format="ubj")).hexdigest(),
                       "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest()}
        saved_learner = json.loads(model.save_raw(raw_format="json"))["learner"]
        saved = saved_learner["gradient_booster"]["model"]
        if args.booster == "gblinear":
            measurement["linear_weights"] = saved["weights"]
            measurement["linear_base_score"] = json.loads(saved_learner["learner_model_param"]["base_score"])
            measurement["prediction_sample"] = prediction[:16].tolist()
            measurement["boosted_rounds"] = saved["boosted_rounds"]
        else:
            trees = saved["trees"]
            leaves = [sum(child < 0 for child in tree["left_children"]) for tree in trees]
            measurement["tree_nodes"] = sum(len(tree["left_children"]) for tree in trees)
            measurement["tree_leaves"] = sum(leaves)
            measurement["max_tree_leaves"] = max(leaves)
        if is_metal:
            profile = ctypes.c_char_p()
            status = xgb.core._LIB.XGBoosterMetalProfile(model.handle, ctypes.byref(profile))
            if status:
                raise RuntimeError(xgb.core._LIB.XGBGetLastError().decode())
            measurement["gpu_profile"] = json.loads(profile.value)
        measurements.append(measurement)
    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(), "build": build, "parameters": params, **provenance,
        "benchmark_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": args.rows, "features": args.features, "rounds": args.rounds,
        "feature_generation": feature_generation, "response_generation": response_generation,
        "weight_generation": weight_generation,
        "output_groups": output_groups,
        "trees": 0 if args.booster == "gblinear" else len(trees),
        "loss_name": loss_name, "constant_baseline_loss": baseline_loss,
        "constant_baseline_method": {
            "reg:squarederror": "Weighted mean label per output" if args.weighted else "Mean label per output" if args.targets > 1 else "Mean label",
            "reg:absoluteerror": "Median label",
            "reg:quantileerror": "Inverse-CDF label quantiles",
            "reg:expectileerror": "Linked one-step stump initialization",
            "reg:normal": "Mean label and log variance with the reference noise floor",
            "survival:aft": "Reference constant time intercept of 0.5",
            "survival:cox": "Reference constant hazard ratio of 1",
            "rank:ndcg": "Stable input order from a zero ranking margin",
            "rank:map": "Stable input order from a zero ranking margin",
            "rank:pairwise": "Stable input order from a zero ranking margin",
        }[args.objective],
        "warmup": {"rows": warmup_rows, "rounds": 1, "prediction": True},
        "measurements": measurements,
        "metal_validation": {key: os.environ.get(key) for key in ('MTL_DEBUG_LAYER', 'MTL_SHADER_VALIDATION')},
        "metal_quantile_controls": {key: os.environ.get(key) for key in ('METALGB_SERIAL_SKETCH', 'METALGB_SERIAL_CUTS', 'METALGB_SERIAL_SUMMARY', 'METALGB_FORCE_PARALLEL_SUMMARY')},
        "median_train_seconds": statistics.median(m["train_seconds"] for m in measurements),
        "median_predict_seconds": statistics.median(m["predict_seconds"] for m in measurements),
        "note": "Measured baseline, not a claim of reaching the M5 Max performance limit.",
    }
    if args.save_result:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model_path = args.output.with_suffix(".model.json")
        prediction_path = args.output.with_suffix(".predictions.npy")
        model.save_model(model_path)
        np.save(prediction_path, prediction)
        artifacts = [("model", model_path), ("predictions", prediction_path)]
        if args.booster == "gbtree":
            cut_path = args.output.with_suffix(".cuts.npz")
            cut_ptr, cut_values = data.get_quantile_cut()
            np.savez(cut_path, indptr=cut_ptr, values=cut_values)
            artifacts.append(("cuts", cut_path))
        report["result_artifacts"] = {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            for name, path in artifacts}
    text = json.dumps(report, indent=2) + "\n"
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)

if __name__ == "__main__":
    main()
