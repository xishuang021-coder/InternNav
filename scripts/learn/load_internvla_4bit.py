from pathlib import Path

import bitsandbytes as bnb
import torch
from transformers import AutoProcessor, BitsAndBytesConfig

from internnav.model.basemodel.internvla_n1.internvla_n1 import (
    InternVLAN1ForCausalLM,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path.home() / "models" / "InternVLA-N1-DualVLN"

DEPTH_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "depth_anything_v2_metric_hypersim_vits.pth"
)

# 加载前检查文件，避免因为路径错误而浪费时间。
assert MODEL_PATH.is_dir(), f"找不到主模型目录：{MODEL_PATH}"
assert DEPTH_PATH.is_file(), f"找不到深度模型：{DEPTH_PATH}"

shards = sorted(MODEL_PATH.glob("model-*.safetensors"))
assert len(shards) == 4, f"主模型应该有4个分片，实际找到：{len(shards)}"

print("找到4个主模型分片")
print("深度模型存在：", DEPTH_PATH)
print("GPU：", torch.cuda.get_device_name(0))

# NF4：将模型中的线性层权重压缩为4-bit。
# 双重量化：进一步压缩量化参数。
# 计算仍使用BF16，与官方模型的数据类型保持一致。
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("\n1/2 正在加载图像和文字预处理器……")

processor = AutoProcessor.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

print("2/2 正在将InternVLA-N1量化并加载到GPU……")
print("这个过程可能需要几分钟，请不要中断。\n")

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

# 统计真正被转换成4-bit的线性层。
linear4bit_count = sum(
    isinstance(module, bnb.nn.Linear4bit)
    for module in model.modules()
)

allocated = torch.cuda.memory_allocated(0) / 1024**3
reserved = torch.cuda.memory_reserved(0) / 1024**3
peak = torch.cuda.max_memory_allocated(0) / 1024**3
footprint = model.get_memory_footprint() / 1024**3

print("\n========== 加载结果 ==========")
print("模型类型：", type(model).__name__)
print("处理器类型：", type(processor).__name__)
print("4-bit线性层数量：", linear4bit_count)
print(f"模型内存占用估计：{footprint:.2f} GiB")
print(f"当前已分配显存：{allocated:.2f} GiB")
print(f"当前已保留显存：{reserved:.2f} GiB")
print(f"加载期间峰值显存：{peak:.2f} GiB")

if linear4bit_count == 0:
    raise RuntimeError("没有发现4-bit线性层，量化没有生效")

print("\nInternVLA-N1 4-bit加载成功。")
input("按回车退出并释放显存……")