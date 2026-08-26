"""Run one offline InternVLA-N1 dual-system sample with NF4 quantization."""

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    InternVLAN1ForCausalLM,
)
from internnav.model.utils.vln_utils import split_and_clean, traj_to_actions


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_MODEL_PATH = Path.home() / "models" / "InternVLA-N1-DualVLN"
DEFAULT_SAMPLE_DIR = REPO_ROOT / "assets" / "realworld_sample_data1"
SYSTEM1_BF16_MODULES = [
    "traj_dit",
    "action_encoder",
    "action_decoder",
    "cond_projector",
    "rgb_model",
    "memory_encoder",
    "rgb_resampler",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sample_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="包含 instruction.txt 和 debug_raw_*.jpg 的样本目录",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="InternVLA-N1 本地模型目录",
    )
    parser.add_argument(
        "--stage",
        choices=("load", "system2", "full"),
        default="system2",
        help="load 只加载并检查模型；system2 只运行视觉语言系统；full 继续运行轨迹系统",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="指定普通 RGB 帧编号，例如 10 对应 debug_raw_0010.jpg",
    )
    parser.add_argument(
        "--with-history",
        action="store_true",
        help="为指定当前帧加入历史普通 RGB 帧",
    )
    parser.add_argument(
        "--follow-look-down",
        action="store_true",
        help="第一轮 System 2 输出恰好为↓时，追加一轮俯视图片对话",
    )
    parser.add_argument(
        "--num-history",
        type=int,
        default=8,
        help="最多选择的历史普通 RGB 帧数量",
    )
    parser.add_argument(
        "--num-sample-trajs",
        type=int,
        default=32,
        help="System 1 候选轨迹数量；默认值来自核心代码",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=10,
        help="System 1 扩散采样步数；默认值来自核心代码",
    )
    return parser.parse_args()


def print_tensor_info(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}"
    )


def print_peak_memory(label):
    peak = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"{label}峰值显存: {peak:.2f} GiB")


def load_model(model_path):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=SYSTEM1_BF16_MODULES,
    )

    print("正在加载 processor：它负责把图片和文字整理成模型输入……")
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "left"

    print("正在以 NF4 4-bit 加载模型，不使用 CPU offload……")
    model = InternVLAN1ForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def inspect_system1_modules(model):
    base_model = model.get_model()
    for module_name in SYSTEM1_BF16_MODULES:
        try:
            module = base_model.get_submodule(module_name)
        except AttributeError as error:
            raise AttributeError(
                f"找不到 System 1 模块：{module_name}"
            ) from error
        parameters = list(module.parameters())
        dtypes = {parameter.dtype for parameter in parameters}
        parameter_count = sum(parameter.numel() for parameter in parameters)
        has_uint8 = torch.uint8 in dtypes
        print(f"System 1 模块: {module_name}")
        print(f"  模块类型: {type(module).__name__}")
        print(f"  参数 dtype 集合: {dtypes}")
        print(f"  参数数量: {parameter_count}")
        print(f"  是否出现 torch.uint8: {has_uint8}")


VALID_ACTIONS = {"STOP", "↑", "←", "→", "↓"}


def classify_system2_output(output):
    """Classify output before deciding whether System 1 can run."""
    if re.search(r"\d", output):
        coordinates = [int(value) for value in re.findall(r"\d+", output)]
        if len(coordinates) < 2:
            return "unknown", None
        # Keep the official Agent's coordinate order: [second number, first].
        return "coordinate", [int(coordinates[1]), int(coordinates[0])]

    if output and re.fullmatch(r"(?:STOP|[↑←→↓\s])+", output):
        return "action", output

    return "unknown", None


def build_system2_messages(images, instruction, with_history=False):
    prompt = (
        "You are an autonomous navigation assistant. "
        "Your task is to <instruction>. "
        "Where should you go next to stay on track? "
        "Please output the next waypoint's coordinates in the image. "
        "Please output STOP when you have successfully completed the task."
    )
    prompt = prompt.replace("<instruction>.", instruction.strip())
    if with_history:
        history_placeholder = "<image>\n" * (len(images) - 1)
        prompt += f" These are your historical observations: {history_placeholder}."
    prompt += " you can see <image>."

    parts = split_and_clean(prompt)
    content = []
    image_index = 0
    for part in parts:
        if part == "<image>":
            content.append({"type": "image", "image": images[image_index]})
            image_index += 1
        else:
            content.append({"type": "text", "text": part})
    if image_index != len(images):
        raise ValueError(
            f"System 2 图片占位符数量与图片数量不一致："
            f"{image_index} != {len(images)}"
        )

    return [{"role": "user", "content": content}]


def build_inputs(
    processor,
    images,
    instruction,
    device,
    with_history=False,
    messages=None,
):
    if messages is None:
        messages = build_system2_messages(images, instruction, with_history)
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[chat_text],
        images=images,
        return_tensors="pt",
    ).to(device)


def run(args):
    if args.num_history < 1:
        raise ValueError(f"--num-history 必须至少为 1：{args.num_history}")
    if args.with_history and args.frame_index is None:
        raise ValueError("指定 --with-history 时必须同时指定 --frame-index")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU；本脚本不会自动改用 CPU。")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"找不到模型目录：{args.model_path}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    processor, model = load_model(args.model_path)
    print_peak_memory("模型加载")
    if args.stage == "load":
        inspect_system1_modules(model)
        return

    if not args.sample_dir.is_dir():
        raise FileNotFoundError(f"找不到样本目录：{args.sample_dir}")

    instruction_path = args.sample_dir / "instruction.txt"
    if not instruction_path.is_file():
        raise FileNotFoundError(f"找不到指令文件：{instruction_path}")
    rgb_paths = sorted(
        path
        for path in args.sample_dir.glob("debug_raw_[0-9]*.jpg")
        if "_look_down" not in path.stem
    )
    if not rgb_paths:
        raise FileNotFoundError(f"找不到普通 RGB 帧：{args.sample_dir}")

    instruction = instruction_path.read_text(encoding="utf-8").strip()
    if args.frame_index is not None:
        if args.frame_index < 0:
            raise ValueError(f"帧编号不能小于 0：{args.frame_index}")
        image_path = args.sample_dir / f"debug_raw_{args.frame_index:04d}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"找不到指定普通 RGB 帧：{image_path}")
    else:
        image_path = rgb_paths[0]
    image_path = image_path.resolve()
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((640, 480), Image.Resampling.LANCZOS)
    current_frame_index = int(image_path.stem.removeprefix("debug_raw_"))
    history_paths = []
    if args.with_history:
        indexed_rgb_paths = []
        for path in rgb_paths:
            frame_index = int(path.stem.removeprefix("debug_raw_"))
            if frame_index < current_frame_index:
                indexed_rgb_paths.append((frame_index, path))
        history_count = min(args.num_history, len(indexed_rgb_paths))
        if history_count:
            sample_positions = np.unique(
                np.linspace(
                    0,
                    len(indexed_rgb_paths) - 1,
                    history_count,
                    dtype=np.int32,
                )
            )
            history_paths = [indexed_rgb_paths[position][1] for position in sample_positions]
    history_images = [
        Image.open(path).convert("RGB") for path in history_paths
    ]
    for history_image in history_images:
        history_image.thumbnail((640, 480), Image.Resampling.LANCZOS)
    input_images = history_images + [image]
    depth = np.full((image.height, image.width), 10.0, dtype=np.float32)

    print(f"仓库根目录: {REPO_ROOT}")
    print(f"样本目录: {args.sample_dir}")
    print(f"使用 RGB 帧绝对路径: {image_path}")
    print(f"历史帧数量: {len(history_paths)}")
    print(
        "历史帧编号: "
        f"{[int(path.stem.removeprefix('debug_raw_')) for path in history_paths]}"
    )
    print(f"当前帧编号: {current_frame_index}")
    print(
        "全部图片输入顺序: "
        f"{[str(path.resolve()) for path in history_paths] + [str(image_path)]}"
    )
    print(f"指令: {instruction}")
    print("深度输入: 固定 10 米占位值，这不是真实深度估计。")

    device = torch.device("cuda:0")
    first_messages = build_system2_messages(
        input_images,
        instruction,
        with_history=args.with_history,
    )
    inputs = build_inputs(
        processor,
        input_images,
        instruction,
        device,
        with_history=args.with_history,
        messages=first_messages,
    )
    print_tensor_info("System 2 input_ids", inputs.input_ids)
    print_tensor_info("System 2 pixel_values", inputs.pixel_values)

    torch.cuda.reset_peak_memory_stats()
    print("System 2：用 model.generate 生成方向文字或像素目标……")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            return_dict_in_generate=True,
        )
    output_ids = output_ids.sequences
    new_tokens = output_ids[0, inputs.input_ids.shape[1] :]
    generated_token_count = new_tokens.shape[0]
    output_text = processor.tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()
    print_tensor_info("System 2 output_ids", output_ids)
    print(f"generated_token_count: {generated_token_count}")
    print(f"generated token IDs: {new_tokens.tolist()}")
    print(f"repr(output_text): {output_text!r}")
    print_peak_memory("System 2")

    output_category, pixel_goal = classify_system2_output(output_text)
    print(f"System 2 解析类别: {output_category}")
    system2_output_ids = output_ids
    system2_inputs = inputs
    used_second_coordinate = False
    if output_category == "action":
        if args.follow_look_down and output_text == "↓":
            look_down_path = (
                args.sample_dir / f"debug_raw_{current_frame_index:04d}_look_down.jpg"
            )
            if not look_down_path.is_file():
                raise FileNotFoundError(
                    f"第一轮 System 2 输出为↓，但找不到对应俯视图片：{look_down_path}"
                )
            look_down_path = look_down_path.resolve()
            look_down_image = Image.open(look_down_path).convert("RGB")
            look_down_image.thumbnail((640, 480), Image.Resampling.LANCZOS)
            second_images = input_images + [look_down_image]
            second_messages = first_messages + [
                {"role": "assistant", "content": output_text},
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
            torch.cuda.reset_peak_memory_stats()
            print(f"俯视图片路径: {look_down_path}")
            print("System 2 第二轮：根据俯视图片生成方向文字或像素目标……")
            with torch.inference_mode():
                second_output_ids = model.generate(
                    **second_inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    return_dict_in_generate=True,
                ).sequences
            second_new_tokens = second_output_ids[
                0, second_inputs.input_ids.shape[1] :
            ]
            second_generated_token_count = second_new_tokens.shape[0]
            second_output_text = processor.tokenizer.decode(
                second_new_tokens,
                skip_special_tokens=True,
            ).strip()
            second_category, second_pixel_goal = classify_system2_output(
                second_output_text
            )
            print(f"第二轮 generated token count: {second_generated_token_count}")
            print(f"第二轮 generated token IDs: {second_new_tokens.tolist()}")
            print(f"第二轮原始输出 repr(output_text): {second_output_text!r}")
            print(f"第二轮 System 2 解析类别: {second_category}")
            if second_category == "coordinate":
                print(f"像素目标 [y, x]: {second_pixel_goal}")
                system2_output_ids = second_output_ids
                system2_inputs = second_inputs
                used_second_coordinate = True
            else:
                print("第二轮输出不是坐标；安全跳过 System 1。")
                return
            print_peak_memory("System 2 第二轮")
            if args.stage == "system2":
                print("当前 stage=system2，因此第二轮坐标已获得但不运行 System 1。")
                return
            print("System 1 将使用第二轮 System 2 的 coordinate。")
        if not used_second_coordinate:
            print("System 2 输出离散动作；没有像素坐标，因此跳过System 1。")
            return
    if output_category == "unknown":
        print("System 2 输出无法解析；停止，不进入System 1。")
        return
    print(
        f"像素目标 [y, x]: {second_pixel_goal if used_second_coordinate else pixel_goal}"
    )

    if args.stage == "system2":
        print("当前 stage=system2，因此已获得像素目标但不运行 System 1。")
        return

    if used_second_coordinate:
        print("System 1 使用的是第二轮 coordinate。")
    torch.cuda.reset_peak_memory_stats()
    image_grid_thw = torch.cat(
        [thw.unsqueeze(0) for thw in system2_inputs.image_grid_thw],
        dim=0,
    )
    with torch.inference_mode():
        # generate_latents 把 System 2 的视觉/文字结果变成 System 1 条件。
        traj_latents = model.generate_latents(
            system2_output_ids,
            system2_inputs.pixel_values,
            image_grid_thw,
        )
    print_tensor_info("System 1 latent", traj_latents)
    print_peak_memory("generate_latents")

    rgb_array = np.asarray(image).astype(np.float32) / 255.0
    rgb_224 = np.asarray(Image.fromarray(np.asarray(image)).resize((224, 224)))
    rgb_224 = torch.from_numpy(rgb_224.astype(np.float32) / 255.0)
    depth_224 = np.asarray(Image.fromarray(depth).resize((224, 224)))
    depth_224 = torch.from_numpy(depth_224.astype(np.float32))
    del rgb_array
    images_dp = torch.stack([rgb_224, rgb_224]).unsqueeze(0).to(device)
    depths_dp = torch.stack([depth_224, depth_224]).unsqueeze(0).unsqueeze(-1).to(device)
    print_tensor_info("System 1 images_dp", images_dp)
    print_tensor_info("System 1 depths_dp", depths_dp)

    torch.cuda.reset_peak_memory_stats()
    print("System 1：用 generate_traj 生成候选轨迹……")
    with torch.inference_mode():
        trajectories = model.generate_traj(
            traj_latents,
            images_dp,
            depths_dp,
            num_inference_steps=args.num_inference_steps,
            num_sample_trajs=args.num_sample_trajs,
        )
    print_tensor_info("System 1 trajectory output", trajectories)
    print_peak_memory("System 1")
    discrete_actions = traj_to_actions(trajectories.clone())
    print(f"轨迹转换后的离散动作: {discrete_actions}")
    print("full 阶段完成：以上轨迹由 System 1 生成，并已转换为动作。")


def main():
    args = parse_args()
    try:
        run(args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        print(
            "CUDA 显存不足：已清空 CUDA 缓存并安全退出。"
            "可先降低 --num-sample-trajs，再重试。"
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()