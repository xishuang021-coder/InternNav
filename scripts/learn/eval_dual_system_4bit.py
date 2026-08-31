"""Offline batch evaluation for the InternVLA-N1 dual-system 4-bit pipeline."""

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_inference_helpers():
    from scripts.learn.infer_dual_system_4bit import (
        DEFAULT_SAMPLE_DIR,
        build_inputs,
        build_system2_messages,
        classify_system2_output,
        load_model,
        traj_to_actions,
    )

    return {
        "DEFAULT_SAMPLE_DIR": DEFAULT_SAMPLE_DIR,
        "build_inputs": build_inputs,
        "build_system2_messages": build_system2_messages,
        "classify_system2_output": classify_system2_output,
        "load_model": load_model,
        "traj_to_actions": traj_to_actions,
    }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_frame_indices(raw: Optional[str]):
    if raw is None:
        return None
    parsed = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        parsed.append(int(value))
    return parsed if parsed else None


def list_sample_frame_indices(sample_dir: Path, frame_indices=None, all_paired_frames=False):
    rgb_paths = sorted(
        path
        for path in sample_dir.glob("debug_raw_[0-9]*.jpg")
        if "_look_down" not in path.stem
    )
    all_indices = []
    for path in rgb_paths:
        try:
            frame_index = int(path.stem.removeprefix("debug_raw_"))
        except ValueError:
            continue
        if all_paired_frames:
            look_down_path = sample_dir / f"debug_raw_{frame_index:04d}_look_down.jpg"
            if not look_down_path.is_file():
                continue
        all_indices.append(frame_index)

    if frame_indices is not None:
        requested = set(frame_indices)
        all_indices = [frame_index for frame_index in all_indices if frame_index in requested]
        missing = sorted(requested - set(all_indices))
        if missing:
            raise ValueError(f"指定 frame-indices 中没有可评测的样本：{missing}")

    return sorted(set(all_indices))


def load_resume_completed(results_path: Path):
    completed = set()
    if not results_path.exists():
        return completed
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "success":
                frame_index = record.get("frame_index")
                if frame_index is not None:
                    completed.add(int(frame_index))
    return completed


def append_result(results_path: Path, record):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_sample_record(scene, frame_index, seed, first_output, second_output, parse_type,
                       pixel_x=None, pixel_y=None, coordinate_in_bounds=None,
                       latent_shape=None, trajectory_shape=None, actions=None,
                       action_count=None, system2_time=0.0, system1_time=0.0,
                       peak_memory=0.0, status="success", error=None):
    return {
        "scene": scene,
        "frame_index": int(frame_index),
        "seed": int(seed),
        "first_round_system2_raw_output": first_output,
        "second_round_system2_raw_output": second_output,
        "parse_type": parse_type,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "coordinate_in_bounds": coordinate_in_bounds,
        "latent_shape": latent_shape,
        "trajectory_shape": trajectory_shape,
        "actions": actions,
        "action_count": int(action_count) if action_count is not None else None,
        "system2_time_sec": float(system2_time),
        "system1_time_sec": float(system1_time),
        "peak_memory_gib": float(peak_memory),
        "status": status,
        "error": error,
    }


def _run_single_eval_frame(sample_dir: Path, frame_index: int, args, processor, model):
    helpers = _import_inference_helpers()
    build_inputs = helpers["build_inputs"]
    build_system2_messages = helpers["build_system2_messages"]
    classify_system2_output = helpers["classify_system2_output"]
    traj_to_actions = helpers["traj_to_actions"]
    instruction_path = sample_dir / "instruction.txt"
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    image_path = sample_dir / f"debug_raw_{frame_index:04d}.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(f"未找到当前帧图像：{image_path}")

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((640, 480), Image.Resampling.LANCZOS)
    rgb_paths = sorted(
        path for path in sample_dir.glob("debug_raw_[0-9]*.jpg") if "_look_down" not in path.stem
    )
    history_paths = []
    if args.with_history:
        indexed_paths = []
        for path in rgb_paths:
            idx = int(path.stem.removeprefix("debug_raw_"))
            if idx < frame_index:
                indexed_paths.append((idx, path))
        history_count = min(args.num_history, len(indexed_paths))
        if history_count:
            sample_positions = np.unique(
                np.linspace(0, len(indexed_paths) - 1, history_count, dtype=np.int32)
            )
            history_paths = [indexed_paths[int(position)][1] for position in sample_positions]
    history_images = [Image.open(path).convert("RGB") for path in history_paths]
    for history_image in history_images:
        history_image.thumbnail((640, 480), Image.Resampling.LANCZOS)
    input_images = history_images + [image]
    device = torch.device("cuda:0")

    first_messages = build_system2_messages(input_images, instruction, with_history=args.with_history)
    first_inputs = build_inputs(
        processor,
        input_images,
        instruction,
        device,
        with_history=args.with_history,
        messages=first_messages,
    )

    torch.cuda.reset_peak_memory_stats()
    system2_start = time.perf_counter()
    with torch.inference_mode():
        first_output_ids = model.generate(
            **first_inputs,
            max_new_tokens=128,
            do_sample=False,
            return_dict_in_generate=True,
        ).sequences
    first_new_tokens = first_output_ids[0, first_inputs.input_ids.shape[1] :]
    first_output_text = processor.tokenizer.decode(first_new_tokens, skip_special_tokens=True).strip()
    first_category, pixel_goal = classify_system2_output(first_output_text)
    second_output_text = ""
    second_category = "unknown"
    second_pixel_goal = None
    chosen_output_ids = first_output_ids
    chosen_inputs = first_inputs
    chosen_pixel_goal = pixel_goal
    chosen_parse_type = first_category

    if first_category == "action" and args.follow_look_down and first_output_text == "↓":
        look_down_path = sample_dir / f"debug_raw_{frame_index:04d}_look_down.jpg"
        if not look_down_path.is_file():
            raise FileNotFoundError(f"第一轮 System 2 输出为↓，但找不到俯视图：{look_down_path}")
        look_down_image = Image.open(look_down_path).convert("RGB")
        look_down_image.thumbnail((640, 480), Image.Resampling.LANCZOS)
        second_images = input_images + [look_down_image]
        second_messages = first_messages + [
            {"role": "assistant", "content": first_output_text},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "you can see "},
                    {"type": "image", "image": look_down_image},
                    {"type": "text", "text": "."},
                ],
            },
        ]
        second_inputs = build_inputs(
            processor,
            second_images,
            instruction,
            device,
            messages=second_messages,
        )
        with torch.inference_mode():
            second_output_ids = model.generate(
                **second_inputs,
                max_new_tokens=128,
                do_sample=False,
                return_dict_in_generate=True,
            ).sequences
        second_new_tokens = second_output_ids[0, second_inputs.input_ids.shape[1] :]
        second_output_text = processor.tokenizer.decode(second_new_tokens, skip_special_tokens=True).strip()
        second_category, second_pixel_goal = classify_system2_output(second_output_text)
        if second_category == "coordinate":
            chosen_output_ids = second_output_ids
            chosen_inputs = second_inputs
            chosen_pixel_goal = second_pixel_goal
            chosen_parse_type = "coordinate"
        else:
            chosen_parse_type = "action"
            chosen_pixel_goal = None
    system2_time = time.perf_counter() - system2_start

    if first_category == "unknown" and second_category == "unknown":
        raise ValueError(f"System 2 输出无法解析：{first_output_text!r} | {second_output_text!r}")

    if chosen_parse_type == "coordinate" and chosen_pixel_goal is None:
        raise ValueError("选定的 System 2 输出没有可用像素坐标。")

    if chosen_parse_type == "coordinate":
        pixel_x = int(chosen_pixel_goal[1])
        pixel_y = int(chosen_pixel_goal[0])
        coordinate_in_bounds = 0 <= pixel_x < image.width and 0 <= pixel_y < image.height
    else:
        pixel_x = None
        pixel_y = None
        coordinate_in_bounds = None

    if args.stage == "system2":
        record = build_sample_record(
            scene=sample_dir.name,
            frame_index=frame_index,
            seed=args.seed,
            first_output=first_output_text,
            second_output=second_output_text,
            parse_type=chosen_parse_type,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            coordinate_in_bounds=coordinate_in_bounds,
            latent_shape=None,
            trajectory_shape=None,
            actions=None,
            action_count=None,
            system2_time=system2_time,
            system1_time=0.0,
            peak_memory=torch.cuda.max_memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0,
            status="success",
            error=None,
        )
        return record

    image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in chosen_inputs.image_grid_thw], dim=0)
    start_system1 = time.perf_counter()
    with torch.inference_mode():
        traj_latents = model.generate_latents(
            chosen_output_ids,
            chosen_inputs.pixel_values,
            image_grid_thw,
        )
    latent_shape = list(traj_latents.shape)

    rgb_array = np.asarray(image).astype(np.float32) / 255.0
    rgb_224 = np.asarray(Image.fromarray(np.asarray(image)).resize((224, 224)))
    rgb_224 = torch.from_numpy(rgb_224.astype(np.float32) / 255.0)
    depth = np.full((image.height, image.width), 10.0, dtype=np.float32)
    depth_224 = np.asarray(Image.fromarray(depth).resize((224, 224)))
    depth_224 = torch.from_numpy(depth_224.astype(np.float32))
    del rgb_array

    images_dp = torch.stack([rgb_224, rgb_224]).unsqueeze(0).to(device)
    depths_dp = torch.stack([depth_224, depth_224]).unsqueeze(0).unsqueeze(-1).to(device)

    with torch.inference_mode():
        trajectories = model.generate_traj(
            traj_latents,
            images_dp,
            depths_dp,
            num_inference_steps=args.num_inference_steps,
            num_sample_trajs=args.num_sample_trajs,
        )
    system1_time = time.perf_counter() - start_system1
    trajectory_shape = list(trajectories.shape)
    if not bool(torch.isfinite(trajectories).all().item()):
        raise ValueError("trajectory contains NaN or Inf")

    actions = traj_to_actions(trajectories.clone())
    action_count = len(actions)
    peak_memory = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

    record = build_sample_record(
        scene=sample_dir.name,
        frame_index=frame_index,
        seed=args.seed,
        first_output=first_output_text,
        second_output=second_output_text,
        parse_type=chosen_parse_type,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        coordinate_in_bounds=coordinate_in_bounds,
        latent_shape=latent_shape,
        trajectory_shape=trajectory_shape,
        actions=actions,
        action_count=action_count,
        system2_time=system2_time,
        system1_time=system1_time,
        peak_memory=peak_memory,
        status="success",
        error=None,
    )
    return record


def write_summary(results_path: Path, summary_path: Path, output_dir: Path):
    records = []
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    summary = {
        "total": len(records),
        "success": sum(1 for item in records if item.get("status") == "success"),
        "error": sum(1 for item in records if item.get("status") == "error"),
        "frame_indices": sorted(int(item.get("frame_index")) for item in records if item.get("frame_index") is not None),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    fieldnames = [
        "scene",
        "frame_index",
        "seed",
        "status",
        "parse_type",
        "pixel_x",
        "pixel_y",
        "coordinate_in_bounds",
        "action_count",
        "system2_time_sec",
        "system1_time_sec",
        "peak_memory_gib",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in records:
            row = {key: item.get(key) for key in fieldnames}
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sample_dir",
        nargs="?",
        type=Path,
        default=None,
        help="包含 instruction.txt 和 debug_raw_*.jpg 的样本目录",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path.home() / "models" / "InternVLA-N1-DualVLN",
        help="InternVLA-N1 本地模型目录",
    )
    parser.add_argument(
        "--stage",
        choices=("load", "system2", "full"),
        default="full",
        help="评测阶段；本批处理默认 full",
    )
    parser.add_argument(
        "--frame-indices",
        type=str,
        default=None,
        help="评测帧编号列表，例如 31,59,89",
    )
    parser.add_argument(
        "--all-paired-frames",
        action="store_true",
        help="只评测同时存在普通 RGB 和俯视图的帧",
    )
    parser.add_argument(
        "--with-history",
        action="store_true",
        help="为当前帧加入历史普通 RGB 帧",
    )
    parser.add_argument(
        "--num-history",
        type=int,
        default=8,
        help="最多选择的历史普通 RGB 帧数量",
    )
    parser.add_argument(
        "--follow-look-down",
        action="store_true",
        help="第一轮 System 2 输出恰好为↓时，追加一轮俯视图片对话",
    )
    parser.add_argument(
        "--num-sample-trajs",
        type=int,
        default=1,
        help="System 1 候选轨迹数量",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=5,
        help="System 1 扩散采样步数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="固定随机种子",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./eval_outputs") / "dual_system_4bit",
        help="评测结果输出目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将评测的编号与配置，不加载模型",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过 results.jsonl 中已成功完成的 frame_index",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_dir is None:
        args.sample_dir = REPO_ROOT / "assets" / "realworld_sample_data1"
    if args.num_history < 1:
        raise ValueError(f"--num-history 必须至少为 1：{args.num_history}")
    if args.with_history and args.frame_indices is None and not args.all_paired_frames:
        pass

    set_seed(args.seed)
    if not args.sample_dir.is_dir():
        raise FileNotFoundError(f"找不到样本目录：{args.sample_dir}")

    frame_indices = parse_frame_indices(args.frame_indices)
    selected_frames = list_sample_frame_indices(
        args.sample_dir,
        frame_indices=frame_indices,
        all_paired_frames=args.all_paired_frames,
    )
    if not selected_frames:
        raise ValueError("没有任何有效的评测帧可供处理。")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    if args.resume:
        completed = load_resume_completed(results_path)
        selected_frames = [frame_index for frame_index in selected_frames if frame_index not in completed]
        if not selected_frames:
            print(f"已跳过所有 frame_index（{len(completed)} 个已完成）")
            write_summary(results_path, summary_path, output_dir)
            return

    if args.dry_run:
        print("Dry-run configuration:")
        print(f"  sample_dir={args.sample_dir}")
        print(f"  stage={args.stage}")
        print(f"  frame_indices={selected_frames}")
        print(f"  all_paired_frames={args.all_paired_frames}")
        print(f"  with_history={args.with_history}")
        print(f"  num_history={args.num_history}")
        print(f"  follow_look_down={args.follow_look_down}")
        print(f"  num_sample_trajs={args.num_sample_trajs}")
        print(f"  num_inference_steps={args.num_inference_steps}")
        print(f"  seed={args.seed}")
        print(f"  output_dir={output_dir}")
        write_summary(results_path, summary_path, output_dir)
        return

    helpers = _import_inference_helpers()
    load_model = helpers["load_model"]
    processor, model = load_model(args.model_path)
    try:
        for frame_index in selected_frames:
            try:
                record = _run_single_eval_frame(args.sample_dir, frame_index, args, processor, model)
            except Exception as exc:
                traceback.print_exc()
                error_text = traceback.format_exc()
                record = build_sample_record(
                    scene=args.sample_dir.name,
                    frame_index=frame_index,
                    seed=args.seed,
                    first_output="",
                    second_output="",
                    parse_type="error",
                    pixel_x=None,
                    pixel_y=None,
                    coordinate_in_bounds=None,
                    latent_shape=None,
                    trajectory_shape=None,
                    actions=None,
                    action_count=None,
                    system2_time=0.0,
                    system1_time=0.0,
                    peak_memory=torch.cuda.max_memory_allocated(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}\n{error_text}",
                )
            append_result(results_path, record)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_summary(results_path, summary_path, output_dir)


if __name__ == "__main__":
    main()
