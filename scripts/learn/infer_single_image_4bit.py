from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    InternVLAN1ForCausalLM,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path.home() / "models" / "InternVLA-N1-DualVLN"

IMAGE_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "iros_challenge"
    / "onsite_competition"
    / "captures"
    / "rs_rgb.jpg"
)

assert MODEL_PATH.is_dir(), f"找不到模型：{MODEL_PATH}"
assert IMAGE_PATH.is_file(), f"找不到图片：{IMAGE_PATH}"

# 这是给模型的导航任务。
instruction = "Go straight ahead toward the open area."

prompt = (
    "You are an autonomous navigation assistant. "
    f"Your task is to {instruction} "
    "Where should you go next to stay on track? "
    "Please output the next waypoint's coordinates in the image. "
    "Please output STOP when you have successfully completed the task."
)

# 读取RGB图像，并限制尺寸以降低第一次测试的显存需求。
image = Image.open(IMAGE_PATH).convert("RGB")
original_size = image.size
image.thumbnail((640, 480), Image.Resampling.LANCZOS)

print("原始图像尺寸：", original_size)
print("输入图像尺寸：", image.size)
print("导航指令：", instruction)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("\n正在加载处理器……")
processor = AutoProcessor.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

print("正在以4-bit加载模型……")
model = InternVLAN1ForCausalLM.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
model.eval()

# Qwen-VL格式：一条用户消息中同时包含图片和文字。
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }
]

chat_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[chat_text],
    images=[image],
    return_tensors="pt",
).to("cuda:0")

print("输入token数量：", inputs.input_ids.shape[1])
print("开始生成导航结果……")

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=False,
    )

new_tokens = output_ids[0, inputs.input_ids.shape[1]:]

answer = processor.decode(
    new_tokens,
    skip_special_tokens=True,
).strip()

peak_memory = torch.cuda.max_memory_allocated(0) / 1024**3

print("\n========== 推理结果 ==========")
print("模型输出：", answer)
print(f"推理峰值显存：{peak_memory:.2f} GiB")