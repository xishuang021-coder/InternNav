#!/usr/bin/env python3
"""Summarize offline dual-system evaluation results from results.jsonl."""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path, help="results.jsonl 路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="汇总输出目录",
    )
    return parser.parse_args()


def as_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stats_summary(values):
    vals = [as_float(v) for v in values if v is not None]
    if not vals:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "population_std": 0.0,
        }
    mean_value = sum(vals) / len(vals)
    median_value = statistics.median(vals)
    min_value = min(vals)
    max_value = max(vals)
    std_value = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return {
        "count": len(vals),
        "mean": mean_value,
        "median": median_value,
        "min": min_value,
        "max": max_value,
        "population_std": std_value,
    }


def summarize_values(name, values):
    return {name: stats_summary(values)}


def validate_record(record):
    issues = []
    parse_type = record.get("parse_type")
    status = record.get("status")

    if parse_type == "coordinate":
        if record.get("pixel_x") is None or record.get("pixel_y") is None:
            issues.append("coordinate record missing pixel_x or pixel_y")
        if not isinstance(record.get("coordinate_in_bounds"), bool):
            issues.append("coordinate record has non-bool coordinate_in_bounds")
        if record.get("latent_shape") is None:
            issues.append("coordinate record missing latent_shape")
        if record.get("trajectory_shape") is None:
            issues.append("coordinate record missing trajectory_shape")
        if record.get("actions") is None:
            issues.append("coordinate record missing actions")
        if as_float(record.get("system1_time_sec"), 0.0) <= 0:
            issues.append("coordinate record has non-positive system1_time_sec")
    elif parse_type == "action":
        if record.get("pixel_x") is not None or record.get("pixel_y") is not None:
            issues.append("action record must have pixel_x and pixel_y set to None")
        if record.get("latent_shape") is not None or record.get("trajectory_shape") is not None:
            issues.append("action record must have latent_shape and trajectory_shape set to None")
        if record.get("actions") is None:
            issues.append("action record missing actions")
        if as_float(record.get("system1_time_sec"), 0.0) != 0.0:
            issues.append("action record system1_time_sec must be 0")

    if status == "success" and parse_type == "action":
        actions = record.get("actions")
        action_count = record.get("action_count")
        if actions is None:
            issues.append("success action record missing actions")
        elif action_count is None or len(actions) != int(action_count):
            issues.append("success action record action_count mismatch")
    if status == "success" and parse_type == "coordinate":
        actions = record.get("actions")
        action_count = record.get("action_count")
        if actions is None:
            issues.append("success coordinate record missing actions")
        elif action_count is None or len(actions) != int(action_count):
            issues.append("success coordinate record action_count mismatch")

    return issues


def main():
    args = parse_args()
    input_path = args.results_jsonl
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"results.jsonl not found: {input_path}")

    valid_records = []
    invalid_json_lines = []
    duplicate_keys = {}
    seen_keys = set()
    first_output_counts = Counter()

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                invalid_json_lines.append({
                    "line_number": line_number,
                    "error": str(exc),
                    "content": stripped[:200],
                })
                continue

            scene = record.get("scene")
            frame_index = record.get("frame_index")
            seed = record.get("seed")
            key = (scene, frame_index, seed)
            if key in seen_keys:
                duplicate_keys.setdefault(key, 0)
                duplicate_keys[key] += 1
            else:
                seen_keys.add(key)

            valid_records.append(record)
            first_output = record.get("first_round_system2_raw_output")
            first_output_counts[str(first_output)] += 1

    total = len(valid_records)
    success_count = sum(1 for record in valid_records if record.get("status") == "success")
    error_count = sum(1 for record in valid_records if record.get("status") == "error")
    consistency_error_count = sum(1 for record in valid_records if record.get("status") == "consistency_error")
    coordinate_count = sum(1 for record in valid_records if record.get("parse_type") == "coordinate")
    direct_action_count = sum(1 for record in valid_records if record.get("parse_type") == "action")
    coordinate_in_bounds_count = sum(
        1
        for record in valid_records
        if record.get("parse_type") == "coordinate" and isinstance(record.get("coordinate_in_bounds"), bool) and record.get("coordinate_in_bounds") is True
    )

    program_success_rate = (success_count / total) if total else 0.0
    coordinate_rate = (coordinate_count / total) if total else 0.0
    direct_action_rate = (direct_action_count / total) if total else 0.0
    coordinate_in_bounds_rate = (coordinate_in_bounds_count / coordinate_count) if coordinate_count else 0.0

    system1_completed_count = sum(
        1
        for record in valid_records
        if record.get("status") == "success"
        and record.get("parse_type") == "coordinate"
        and as_float(record.get("system1_time_sec"), 0.0) > 0.0
    )
    system1_completed_rate = (system1_completed_count / total) if total else 0.0

    successful_system2_times = [
        as_float(record.get("system2_time_sec"))
        for record in valid_records
        if record.get("status") == "success"
    ]
    coordinate_system2_times = [
        as_float(record.get("system2_time_sec"))
        for record in valid_records
        if record.get("parse_type") == "coordinate" and record.get("status") == "success"
    ]
    direct_action_system2_times = [
        as_float(record.get("system2_time_sec"))
        for record in valid_records
        if record.get("parse_type") == "action" and record.get("status") == "success"
    ]
    system1_completed_times = [
        as_float(record.get("system1_time_sec"))
        for record in valid_records
        if record.get("status") == "success"
        and record.get("parse_type") == "coordinate"
        and as_float(record.get("system1_time_sec"), 0.0) > 0.0
    ]
    core_total_times = [
        as_float(record.get("system2_time_sec"), 0.0) + as_float(record.get("system1_time_sec"), 0.0)
        for record in valid_records
        if record.get("status") == "success"
    ]

    success_peak_memory = [
        as_float(record.get("peak_memory_gib"))
        for record in valid_records
        if record.get("status") == "success"
    ]

    coordinate_action_counts = [
        int(record.get("action_count") or 0)
        for record in valid_records
        if record.get("parse_type") == "coordinate" and record.get("status") == "success"
    ]
    direct_action_action_counts = [
        int(record.get("action_count") or 0)
        for record in valid_records
        if record.get("parse_type") == "action" and record.get("status") == "success"
    ]

    summaries = {
        "total": total,
        "success": success_count,
        "error": error_count,
        "program_success_rate": program_success_rate,
        "coordinate_count": coordinate_count,
        "coordinate_rate": coordinate_rate,
        "direct_action_count": direct_action_count,
        "direct_action_rate": direct_action_rate,
        "coordinate_in_bounds_count": coordinate_in_bounds_count,
        "coordinate_in_bounds_rate": coordinate_in_bounds_rate,
        "system1_completed_count": system1_completed_count,
        "system1_completed_rate": system1_completed_rate,
        "consistency_error_count": consistency_error_count,
        "invalid_json_count": len(invalid_json_lines),
        "duplicate_record_keys": duplicate_keys,
        "duplicate_key_count": sum(duplicate_keys.values()),
        "first_round_system2_raw_output_counts": dict(sorted(first_output_counts.items())),
        "time_stats": {
            "all_success_system2_time_sec": stats_summary(successful_system2_times),
            "coordinate_system2_time_sec": stats_summary(coordinate_system2_times),
            "direct_action_system2_time_sec": stats_summary(direct_action_system2_times),
            "system1_completed_system1_time_sec": stats_summary(system1_completed_times),
            "all_success_core_total_time_sec": stats_summary(core_total_times),
        },
        "peak_memory_gib": stats_summary(success_peak_memory),
        "action_count_stats": {
            "trajectory_derived_actions": stats_summary(coordinate_action_counts),
            "direct_actions": stats_summary(direct_action_action_counts),
        },
    }

    # Validate coordinate and action semantics
    validation_issues = []
    for index, record in enumerate(valid_records, start=1):
        issues = validate_record(record)
        if issues:
            validation_issues.append({
                "index": index,
                "scene": record.get("scene"),
                "frame_index": record.get("frame_index"),
                "seed": record.get("seed"),
                "parse_type": record.get("parse_type"),
                "status": record.get("status"),
                "issues": issues,
            })
    summaries["validation_issues"] = validation_issues
    summaries["validation_issue_count"] = len(validation_issues)

    detailed_summary = {
        "input_file": str(input_path),
        "summary": summaries,
    }

    detailed_summary_path = output_dir / "detailed_summary.json"
    detailed_summary_path.write_text(json.dumps(detailed_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "per_frame.csv"
    fieldnames = [
        "scene",
        "frame_index",
        "seed",
        "status",
        "parse_type",
        "first_output",
        "second_output",
        "pixel_x",
        "pixel_y",
        "coordinate_in_bounds",
        "latent_shape",
        "trajectory_shape",
        "actions",
        "action_count",
        "system2_time_sec",
        "system1_time_sec",
        "core_total_time_sec",
        "peak_memory_gib",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in valid_records:
            row = {
                "scene": record.get("scene"),
                "frame_index": record.get("frame_index"),
                "seed": record.get("seed"),
                "status": record.get("status"),
                "parse_type": record.get("parse_type"),
                "first_output": record.get("first_round_system2_raw_output"),
                "second_output": record.get("second_round_system2_raw_output"),
                "pixel_x": record.get("pixel_x"),
                "pixel_y": record.get("pixel_y"),
                "coordinate_in_bounds": record.get("coordinate_in_bounds"),
                "latent_shape": record.get("latent_shape"),
                "trajectory_shape": record.get("trajectory_shape"),
                "actions": record.get("actions"),
                "action_count": record.get("action_count"),
                "system2_time_sec": record.get("system2_time_sec"),
                "system1_time_sec": record.get("system1_time_sec"),
                "core_total_time_sec": as_float(record.get("system2_time_sec"), 0.0) + as_float(record.get("system1_time_sec"), 0.0),
                "peak_memory_gib": record.get("peak_memory_gib"),
                "error": record.get("error"),
            }
            writer.writerow(row)

    report_lines = [
        "# 离线评测结果汇总",
        "",
        "这些是离线程序有效性指标，不是 SR、SPL 或导航成功率。",
        "",
        "- 这些指标用于验证离线评测脚本/记录格式/逻辑一致性。",
        "- 它们不是导航任务自身的成功率指标。",
        "- 当前 17 帧属于同一个 episode，不是 17 个独立任务。",
        "",
        "## 总体统计",
        f"- total: {total}",
        f"- success: {success_count}",
        f"- error: {error_count}",
        f"- consistency_error_count: {consistency_error_count}",
        f"- program_success_rate: {program_success_rate:.4f}",
        f"- coordinate_count: {coordinate_count}",
        f"- coordinate_rate: {coordinate_rate:.4f}",
        f"- direct_action_count: {direct_action_count}",
        f"- direct_action_rate: {direct_action_rate:.4f}",
        f"- coordinate_in_bounds_count: {coordinate_in_bounds_count}",
        f"- coordinate_in_bounds_rate: {coordinate_in_bounds_rate:.4f}",
        f"- system1_completed_count: {system1_completed_count}",
        f"- system1_completed_rate: {system1_completed_rate:.4f}",
        "",
        "## 关键时间统计",
    ]

    for key, value in summaries["time_stats"].items():
        report_lines.append(f"- {key}: count={value['count']}, mean={value['mean']:.6f}, median={value['median']:.6f}, min={value['min']:.6f}, max={value['max']:.6f}, population_std={value['population_std']:.6f}")

    report_lines.extend([
        "",
        "## 显存统计",
        f"- peak_memory_gib: count={summaries['peak_memory_gib']['count']}, mean={summaries['peak_memory_gib']['mean']:.6f}, median={summaries['peak_memory_gib']['median']:.6f}, min={summaries['peak_memory_gib']['min']:.6f}, max={summaries['peak_memory_gib']['max']:.6f}",
        "",
        "## 动作数量统计",
    ])
    for key, value in summaries["action_count_stats"].items():
        report_lines.append(f"- {key}: count={value['count']}, mean={value['mean']:.6f}, median={value['median']:.6f}, min={value['min']:.6f}, max={value['max']:.6f}, population_std={value['population_std']:.6f}")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return detailed_summary_path, csv_path, report_path


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
