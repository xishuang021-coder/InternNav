#!/usr/bin/env python3
"""Compare offline dual-system evaluation results across random seeds using only the stdlib."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

VALID_PARSE_TYPES = {"coordinate", "action"}
REQUIRED_FIELDS = {
    "scene",
    "frame_index",
    "seed",
    "status",
    "parse_type",
    "first_round_system2_raw_output",
    "second_round_system2_raw_output",
    "system2_time_sec",
    "system1_time_sec",
    "peak_memory_gib",
}
COORDINATE_REQUIRED_FIELDS = {"pixel_x", "pixel_y", "coordinate_in_bounds", "actions", "action_count"}
ACTION_REQUIRED_FIELDS = {"actions", "action_count"}
CONFIG_KEYS = [
    "stage",
    "with_history",
    "num_history",
    "follow_look_down",
    "num_sample_trajs",
    "num_inference_steps",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", nargs="+", type=Path, help="两个或多个 results.jsonl 文件路径")
    parser.add_argument("--output-dir", type=Path, required=True, help="比较输出目录")
    return parser.parse_args()


def safe_float(value, default=0.0):
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value, default=0):
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def population_std(values):
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        current = [i]
        for j, bv in enumerate(b, start=1):
            insertions = current[j - 1] + 1
            deletions = previous[j] + 1
            substitutions = previous[j - 1] + (av != bv)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]


def sorted_seed_ids(seed_to_records):
    return sorted(seed_to_records.keys(), key=lambda s: int(s))


def read_jsonl(path):
    records = []
    seen = set()
    seed_values = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: each line must decode to a JSON object")
            for field in REQUIRED_FIELDS:
                if field not in record:
                    raise ValueError(f"{path}:{line_no}: missing required field '{field}'")
            scene = record.get("scene")
            frame_index = record.get("frame_index")
            seed = record.get("seed")
            if scene is None or frame_index is None or seed is None:
                raise ValueError(f"{path}:{line_no}: missing scene/frame_index/seed")
            key = (scene, int(frame_index), int(seed))
            if key in seen:
                raise ValueError(f"{path}:{line_no}: duplicate (scene, frame_index, seed) = {(scene, frame_index, seed)}")
            seen.add(key)
            seed_values.add(int(seed))
            records.append(record)
    if len(seed_values) != 1:
        raise ValueError(f"{path}: expected exactly one seed, found {sorted(seed_values)}")
    seed = next(iter(seed_values))
    for record in records:
        if record.get("seed") != seed:
            raise ValueError(f"{path}: mixed seeds found inside same file: {record.get('seed')} vs {seed}")
        if record.get("status") != "success":
            raise ValueError(f"{path}: record with scene={record.get('scene')} frame_index={record.get('frame_index')} has status {record.get('status')!r}, expected 'success'")
        parse_type = record.get("parse_type")
        if parse_type not in VALID_PARSE_TYPES:
            raise ValueError(f"{path}: record with scene={record.get('scene')} frame_index={record.get('frame_index')} has invalid parse_type {parse_type!r}; allowed {sorted(VALID_PARSE_TYPES)}")
        missing_fields = []
        if parse_type == "coordinate":
            missing_fields = sorted(COORDINATE_REQUIRED_FIELDS - set(record.keys()))
            required = COORDINATE_REQUIRED_FIELDS
        else:
            missing_fields = sorted(ACTION_REQUIRED_FIELDS - set(record.keys()))
            required = ACTION_REQUIRED_FIELDS
        if missing_fields:
            raise ValueError(f"{path}: record with scene={record.get('scene')} frame_index={record.get('frame_index')} missing fields for parse_type '{parse_type}': {missing_fields}")
        for field in required:
            if field not in record:
                raise ValueError(f"{path}: record missing required field '{field}'")
        actions = record.get("actions")
        try:
            action_count = int(record.get("action_count"))
        except (TypeError, ValueError):
            raise ValueError(f"{path}: record action_count is not int-like: {record.get('action_count')!r}")
        if not isinstance(actions, list):
            raise ValueError(f"{path}: record actions must be a list, got {type(actions).__name__}")
        if action_count != len(actions):
            raise ValueError(f"{path}: record with scene={record.get('scene')} frame_index={record.get('frame_index')} has action_count={action_count} but len(actions)={len(actions)}")
        if parse_type == "coordinate":
            if record.get("pixel_x") is None or record.get("pixel_y") is None:
                raise ValueError(f"{path}: coordinate record missing pixel_x or pixel_y")
            if not isinstance(record.get("coordinate_in_bounds"), bool):
                raise ValueError(f"{path}: coordinate record coordinate_in_bounds must be bool, got {type(record.get('coordinate_in_bounds')).__name__}")
    return records, seed


def load_run_config(path):
    config_path = path.parent / "run_config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path}: run_config.json must decode to a JSON object")
    return config


def compare_values(a, b):
    if isinstance(a, list):
        return a == b
    return a == b


def frame_identity_key(record):
    return (record.get("scene"), int(record.get("frame_index")))


def build_seed_index(records):
    index = {}
    for record in records:
        key = frame_identity_key(record)
        index[key] = record
    return index


def summarize_seed_performance(seed_records):
    success_records = seed_records
    system2_times = [safe_float(rec.get("system2_time_sec"), 0.0) for rec in success_records]
    coord_records = [rec for rec in success_records if rec.get("parse_type") == "coordinate"]
    action_records = [rec for rec in success_records if rec.get("parse_type") == "action"]
    coord_times = [safe_float(rec.get("system2_time_sec"), 0.0) for rec in coord_records]
    action_times = [safe_float(rec.get("system2_time_sec"), 0.0) for rec in action_records]
    system1_times = [safe_float(rec.get("system1_time_sec"), 0.0) for rec in success_records if safe_float(rec.get("system1_time_sec"), 0.0) > 0.0]
    core_total_times = [safe_float(rec.get("system2_time_sec"), 0.0) + safe_float(rec.get("system1_time_sec"), 0.0) for rec in success_records]
    peak_memory = max([safe_float(rec.get("peak_memory_gib"), 0.0) for rec in success_records], default=0.0)
    action_counts = [len(rec.get("actions", [])) for rec in coord_records]
    return {
        "seed": int(success_records[0].get("seed")),
        "system2_avg_time_sec": mean(system2_times),
        "coordinate_system2_avg_time_sec": mean(coord_times),
        "direct_action_system2_avg_time_sec": mean(action_times),
        "system1_avg_time_sec": mean(system1_times),
        "core_total_avg_time_sec": mean(core_total_times),
        "max_peak_memory_gib": peak_memory,
        "coordinate_trajectory_action_count_avg": mean(action_counts),
    }


def compare_common_frames(seed_to_index, seed_ids):
    common_keys = set.intersection(*(set(index.keys()) for index in seed_to_index.values()))
    system2_inconsistent = []
    direct_action_inconsistent = []
    coordinate_frame_summary = []
    per_frame_rows = []
    system2_identical_count = 0
    direct_action_identical_count = 0
    coord_identical_count = 0
    coord_metrics = []

    for key in sorted(common_keys):
        records = {seed: seed_to_index[seed][key] for seed in seed_ids}
        parse_types = {seed: records[seed].get("parse_type") for seed in seed_ids}
        system2_fields = {
            "parse_type": parse_types,
            "first_round_system2_raw_output": {seed: records[seed].get("first_round_system2_raw_output") for seed in seed_ids},
            "second_round_system2_raw_output": {seed: records[seed].get("second_round_system2_raw_output") for seed in seed_ids},
            "pixel_x": {seed: records[seed].get("pixel_x") for seed in seed_ids},
            "pixel_y": {seed: records[seed].get("pixel_y") for seed in seed_ids},
            "coordinate_in_bounds": {seed: records[seed].get("coordinate_in_bounds") for seed in seed_ids},
        }
        all_same = all(
            compare_values(system2_fields[field][seed_ids[0]], system2_fields[field][seed])
            for field in system2_fields
            for seed in seed_ids[1:]
        )
        if all_same:
            system2_identical_count += 1
        else:
            system2_inconsistent.append({
                "scene": key[0],
                "frame_index": key[1],
                "details": system2_fields,
            })

        row = {
            "scene": key[0],
            "frame_index": key[1],
            "parse_type": records[seed_ids[0]].get("parse_type"),
            "system2_identical": all_same,
            "direct_actions_identical": False,
            "trajectory_actions_identical": False,
            "seed0_actions": json.dumps(records[seed_ids[0]].get("actions", []), separators=(",", ":")),
            "seed1_actions": json.dumps(records[seed_ids[1]].get("actions", []), separators=(",", ":")),
            "seed2_actions": json.dumps(records[seed_ids[2]].get("actions", []), separators=(",", ":")),
            "seed0_action_count": len(records[seed_ids[0]].get("actions", [])),
            "seed1_action_count": len(records[seed_ids[1]].get("actions", [])),
            "seed2_action_count": len(records[seed_ids[2]].get("actions", [])),
            "action_count_mean": 0.0,
            "action_count_std": 0.0,
            "pairwise_normalized_edit_distance_mean": 0.0,
        }

        action_records = [records[seed] for seed in seed_ids if records[seed].get("parse_type") == "action"]
        if action_records:
            action_seqs = [rec.get("actions", []) for rec in action_records]
            action_same = all(seq == action_seqs[0] for seq in action_seqs[1:])
            action_counts = [len(seq) for seq in action_seqs]
            action_count_same = all(count == action_counts[0] for count in action_counts[1:])
            row["direct_actions_identical"] = action_same and action_count_same
            if not row["direct_actions_identical"]:
                direct_action_inconsistent.append({
                    "scene": key[0],
                    "frame_index": key[1],
                    "seed_actions": {seed: records[seed].get("actions", []) for seed in seed_ids if records[seed].get("parse_type") == "action"},
                    "seed_action_counts": {seed: len(records[seed].get("actions", [])) for seed in seed_ids if records[seed].get("parse_type") == "action"},
                })
            if action_same and action_count_same:
                direct_action_identical_count += 1

        coord_records = [records[seed] for seed in seed_ids if records[seed].get("parse_type") == "coordinate"]
        if coord_records:
            seq_by_seed = {seed: records[seed].get("actions", []) for seed in seed_ids if records[seed].get("parse_type") == "coordinate"}
            seqs = list(seq_by_seed.values())
            action_counts = [len(seq) for seq in seqs]
            pairwise_norm = []
            for i in range(len(seed_ids)):
                for j in range(i + 1, len(seed_ids)):
                    if seed_ids[i] not in seq_by_seed or seed_ids[j] not in seq_by_seed:
                        continue
                    a = seq_by_seed[seed_ids[i]]
                    b = seq_by_seed[seed_ids[j]]
                    pairwise_norm.append(levenshtein(a, b) / max(max(len(a), len(b)), 1))
            mean_norm = mean(pairwise_norm) if pairwise_norm else 0.0
            row["action_count_mean"] = mean(action_counts) if action_counts else 0.0
            row["action_count_std"] = population_std(action_counts) if len(action_counts) > 1 else 0.0
            row["pairwise_normalized_edit_distance_mean"] = mean_norm
            trajectory_same = all(seqs[0] == seq for seq in seqs[1:])
            row["trajectory_actions_identical"] = trajectory_same
            if trajectory_same:
                coord_identical_count += 1
            coord_metrics.append({
                "scene": key[0],
                "frame_index": key[1],
                "action_count_mean": row["action_count_mean"],
                "action_count_std": row["action_count_std"],
                "pairwise_normalized_edit_distance_mean": mean_norm,
                "trajectory_actions_identical": trajectory_same,
                "actions_by_seed": {seed: seq_by_seed[seed] for seed in sorted(seq_by_seed.keys())},
            })
        else:
            row["direct_actions_identical"] = False
            row["trajectory_actions_identical"] = False
        per_frame_rows.append(row)

    total_common = len(common_keys)
    system2_identical_rate = system2_identical_count / total_common if total_common else 0.0
    direct_action_total = sum(1 for row in per_frame_rows if row["parse_type"] == "action")
    direct_action_rate = (direct_action_identical_count / direct_action_total) if direct_action_total else 0.0
    coord_total = sum(1 for row in per_frame_rows if row["parse_type"] == "coordinate")
    coord_identical_rate = (coord_identical_count / coord_total) if coord_total else 0.0

    coordinate_frames = [metric for metric in coord_metrics]
    stable_sorted = sorted(coordinate_frames, key=lambda item: item["pairwise_normalized_edit_distance_mean"])
    least_stable = sorted(coordinate_frames, key=lambda item: item["pairwise_normalized_edit_distance_mean"], reverse=True)

    return {
        "total_common_frames": total_common,
        "system2_identical_frame_count": system2_identical_count,
        "system2_identical_rate": system2_identical_rate,
        "direct_action_identical_frame_count": direct_action_identical_count,
        "direct_action_identical_rate": direct_action_rate,
        "coordinate_identical_frame_count": coord_identical_count,
        "coordinate_identical_rate": coord_identical_rate,
        "system2_inconsistent_frames": system2_inconsistent,
        "direct_action_inconsistent_frames": direct_action_inconsistent,
        "coordinate_varying_frames": coordinate_frames,
        "most_stable_coordinate_frames": stable_sorted[:5],
        "least_stable_coordinate_frames": least_stable[:5],
        "per_frame_rows": per_frame_rows,
    }


def summarize_config_warnings(file_paths):
    warnings = []
    configs = []
    for path in file_paths:
        config = load_run_config(path)
        if config is not None:
            configs.append({"path": str(path), "config": config})
    if not configs:
        return []
    baseline = configs[0]["config"]
    for entry in configs[1:]:
        for key in CONFIG_KEYS:
            if baseline.get(key) != entry["config"].get(key):
                warnings.append({
                    "path_a": str(configs[0]["path"]),
                    "path_b": str(entry["path"]),
                    "key": key,
                    "value_a": baseline.get(key),
                    "value_b": entry["config"].get(key),
                })
    return warnings


def write_summary_json(summary_path, summary):
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(csv_path, rows):
    fieldnames = [
        "scene",
        "frame_index",
        "parse_type",
        "system2_identical",
        "direct_actions_identical",
        "trajectory_actions_identical",
        "seed0_actions",
        "seed1_actions",
        "seed2_actions",
        "seed0_action_count",
        "seed1_action_count",
        "seed2_action_count",
        "action_count_mean",
        "action_count_std",
        "pairwise_normalized_edit_distance_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_report(summary, file_paths, config_warnings):
    total_frames = summary["total_common_frames"]
    program_success_rate = 1.0 if total_frames else 0.0
    report_lines = [
        "# Cross-seed evaluation comparison report",
        "",
        "## 1. Executive summary",
        f"- 输入文件数: {len(file_paths)}",
        f"- 共比较关键帧数: {total_frames}",
        f"- 程序级成功率: {program_success_rate:.4f}",
        "- 说明：这里的程序级成功率反映的是结果文件中所有样本都成功写出且通过结构校验，不是 SR、SPL 或真实导航成功率。",
        "",
        "## 2. System 2 输出稳定性",
        f"- 完全一致关键帧数量: {summary['system2_identical_frame_count']} / {total_frames}",
        f"- 完全一致率: {summary['system2_identical_rate']:.4f}",
        f"- 不一致帧数: {len(summary['system2_inconsistent_frames'])}",
        "- 评估对象包括 parse_type、first_round_system2_raw_output、second_round_system2_raw_output、pixel_x、pixel_y 和 coordinate_in_bounds。",
        "- 以上统计反映的是 System 2 输出稳定性，而不是导航任务成功率。",
        "",
        "## 3. direct action 与 coordinate 动作稳定性",
        f"- direct action 完全一致关键帧数量: {summary['direct_action_identical_frame_count']}",
        f"- direct action 完全一致率: {summary['direct_action_identical_rate']:.4f}",
        f"- coordinate 轨迹动作序列完全一致关键帧数量: {summary['coordinate_identical_frame_count']}",
        f"- coordinate 轨迹动作序列完全一致率: {summary['coordinate_identical_rate']:.4f}",
        "- 这里的稳定性衡量的是 System 1 随机轨迹在跨 seed 下的一致性；它不等同于真实导航成功率，也不是 SR/SPL。",
        "",
        "## 4. 性能与随机性说明",
        "- System 1 轨迹在 coordinate 分支中会受随机种子影响，因此跨 seed 的 action 序列和长度可能不同；这属于随机轨迹稳定性分析，而非任务成功判定。",
        f"- 当前比较样本覆盖 {total_frames} 个共享帧，来自同一个 episode。",
        "",
        "## 5. 配置一致性检查",
    ]
    if not config_warnings:
        report_lines.append("- 未发现 run_config.json 配置差异，或者当前结果目录中没有配置文件。")
    else:
        report_lines.append("- 检测到配置差异：")
        for warning in config_warnings:
            report_lines.append(
                f"  - {warning['path_a']} vs {warning['path_b']}: key '{warning['key']}' differs ({warning['value_a']} != {warning['value_b']})"
            )
    report_lines.extend([
        "",
        "## 6. 结论",
        "- 本报告用于比较不同随机种子下的离线结果稳定性。",
        "- 该指标不代表 SR、SPL 或真实导航成功率；它是 System 2 输出、System 1 轨迹和动作序列的稳定性评估。",
    ])
    return "\n".join(report_lines) + "\n"


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.results_jsonl) < 2:
        raise ValueError("At least two results.jsonl files are required for comparison")

    file_paths = [Path(p) for p in args.results_jsonl]
    for path in file_paths:
        if not path.exists():
            raise FileNotFoundError(f"results.jsonl not found: {path}")

    seed_to_records = {}
    seed_to_index = {}
    seed_ids = []
    for file_path in file_paths:
        records, seed = read_jsonl(file_path)
        seed_to_records[seed] = records
        seed_to_index[seed] = build_seed_index(records)
        seed_ids.append(seed)

    seed_ids = sorted(seed_ids)
    common_keys = set.intersection(*(set(index.keys()) for index in [seed_to_index[s] for s in seed_ids]))
    if not common_keys:
        raise ValueError("No common (scene, frame_index) keys were found across the provided seed files")

    per_frame_comparison = compare_common_frames({seed: seed_to_index[seed] for seed in seed_ids}, seed_ids)
    summary = {
        "input_files": [str(path) for path in file_paths],
        "seed_ids": seed_ids,
        "total_common_frames": per_frame_comparison["total_common_frames"],
        "system2_identical_frame_count": per_frame_comparison["system2_identical_frame_count"],
        "system2_identical_rate": per_frame_comparison["system2_identical_rate"],
        "direct_action_identical_frame_count": per_frame_comparison["direct_action_identical_frame_count"],
        "direct_action_identical_rate": per_frame_comparison["direct_action_identical_rate"],
        "coordinate_identical_frame_count": per_frame_comparison["coordinate_identical_frame_count"],
        "coordinate_identical_rate": per_frame_comparison["coordinate_identical_rate"],
        "system2_inconsistent_frames": per_frame_comparison["system2_inconsistent_frames"],
        "direct_action_inconsistent_frames": per_frame_comparison["direct_action_inconsistent_frames"],
        "coord_stability_summary": {
            "average_pairwise_normalized_edit_distance": mean(
                [
                    item["pairwise_normalized_edit_distance_mean"]
                    for item in per_frame_comparison["coordinate_varying_frames"]
                ]
            ) if per_frame_comparison["coordinate_varying_frames"] else 0.0,
            "most_stable_frames": per_frame_comparison["most_stable_coordinate_frames"],
            "least_stable_frames": per_frame_comparison["least_stable_coordinate_frames"],
            "overall_action_count_mean": mean(
                [
                    count
                    for item in per_frame_comparison["coordinate_varying_frames"]
                    for count in [item["action_count_mean"]]
                ]
            ) if per_frame_comparison["coordinate_varying_frames"] else 0.0,
            "overall_action_count_population_std": population_std(
                [
                    count
                    for item in per_frame_comparison["coordinate_varying_frames"]
                    for count in [item["action_count_mean"]]
                ]
            ) if len(per_frame_comparison["coordinate_varying_frames"]) > 1 else 0.0,
        },
        "performance_by_seed": {},
        "performance_across_seeds": {},
        "config_warnings": [],
    }

    performance_by_seed = {}
    for seed in seed_ids:
        performance_by_seed[seed] = summarize_seed_performance(seed_to_records[seed])
    summary["performance_by_seed"] = performance_by_seed

    per_seed_metrics = [
        [
            metrics["system2_avg_time_sec"],
            metrics["coordinate_system2_avg_time_sec"],
            metrics["direct_action_system2_avg_time_sec"],
            metrics["system1_avg_time_sec"],
            metrics["core_total_avg_time_sec"],
            metrics["max_peak_memory_gib"],
            metrics["coordinate_trajectory_action_count_avg"],
        ]
        for _, metrics in performance_by_seed.items()
    ]
    metric_names = [
        "system2_avg_time_sec",
        "coordinate_system2_avg_time_sec",
        "direct_action_system2_avg_time_sec",
        "system1_avg_time_sec",
        "core_total_avg_time_sec",
        "max_peak_memory_gib",
        "coordinate_trajectory_action_count_avg",
    ]
    performance_across_seeds = {}
    for idx, name in enumerate(metric_names):
        values = [row[idx] for row in per_seed_metrics]
        performance_across_seeds[name] = {
            "mean": mean(values),
            "population_std": population_std(values),
        }
    summary["performance_across_seeds"] = performance_across_seeds

    config_warnings = summarize_config_warnings(file_paths)
    summary["config_warnings"] = config_warnings

    write_summary_json(output_dir / "cross_seed_summary.json", summary)
    write_csv(output_dir / "per_frame_seed_comparison.csv", per_frame_comparison["per_frame_rows"])
    report_text = build_report(summary, file_paths, config_warnings)
    (output_dir / "cross_seed_report.md").write_text(report_text, encoding="utf-8")

    print(f"Saved summary: {output_dir / 'cross_seed_summary.json'}")
    print(f"Saved CSV: {output_dir / 'per_frame_seed_comparison.csv'}")
    print(f"Saved report: {output_dir / 'cross_seed_report.md'}")
    print(f"common_frames={summary['total_common_frames']} system2_identical_rate={summary['system2_identical_rate']:.4f} coordinate_identical_rate={summary['coordinate_identical_rate']:.4f}")


if __name__ == "__main__":
    main()
