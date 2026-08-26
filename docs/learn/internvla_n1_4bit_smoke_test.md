# InternVLA-N1 4-bit 双系统推理冒烟测试

## 1. 测试目标

在单张 NVIDIA GeForce RTX 3080 Ti 12GB 上，以混合精度方式跑通 InternVLA-N1 的完整离线推理链路：

```text
历史 RGB 图像
→ System 2 导航决策
→ 俯视图二次决策
→ 像素目标
→ latent
→ System 1 扩散轨迹
→ 离散动作
```

本测试的目标是验证代码链路能够运行，不是复现论文指标。

## 2. 测试环境

- 操作系统：Ubuntu 20.04.6 LTS
- GPU：NVIDIA GeForce RTX 3080 Ti
- 显存：12GB
- 内存：约 31GB
- NVIDIA 驱动：535.261.03
- 驱动支持的 CUDA 版本：12.2
- Python：3.10.20
- Conda 环境：`/home/hp/miniforge3/envs/internvla-learn`
- Git 分支：`learn/internvla-n1`
- 模型：InternVLA-N1-DualVLN
- 模型本地路径：`/home/hp/models/InternVLA-N1-DualVLN`

## 3. 混合精度加载方案

视觉语言主模型使用：

- NF4 4-bit
- Double Quantization
- BF16 compute dtype
- FlashAttention 2

System 1 以下模块保留 BF16，不进行 4-bit 量化：

```text
traj_dit
action_encoder
action_decoder
cond_projector
rgb_model
memory_encoder
rgb_resampler
```

这样可以避免 PyTorch `MultiheadAttention` 使用 Byte 权重与 BF16 输入相乘时出现 dtype 冲突。

## 4. 测试数据

样本目录：

```text
assets/realworld_sample_data1
```

当前普通 RGB 帧：

```text
debug_raw_0031.jpg
```

配对俯视图：

```text
debug_raw_0031_look_down.jpg
```

历史图片数量：

```text
8
```

## 5. 运行命令

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/hp/miniforge3/envs/internvla-learn/bin/python -u \
scripts/learn/infer_dual_system_4bit.py \
assets/realworld_sample_data1 \
--stage full \
--frame-index 31 \
--with-history \
--num-history 8 \
--follow-look-down \
--num-sample-trajs 1 \
--num-inference-steps 5 \
2>&1 | tee /tmp/internvla_full_mixed_frame0031.log
```

## 6. System 2 结果

第一轮结合历史图片和当前普通图片进行推理，模型要求继续查看俯视图。

第二轮输入配对俯视图后得到：

```text
原始输出：471 208
解析类别：coordinate
像素目标 [y, x]：[208, 471]
```

第二轮 generated token IDs：

```text
[19, 22, 16, 220, 17, 15, 23, 151645]
```

## 7. System 1 结果

System 2 生成的 latent：

```text
shape=(1, 4, 3584)
dtype=torch.bfloat16
device=cuda:0
```

System 1 输入：

```text
images_dp:
shape=(1, 2, 224, 224, 3)
dtype=torch.float32
device=cuda:0

depths_dp:
shape=(1, 2, 224, 224, 1)
dtype=torch.float32
device=cuda:0
```

System 1 生成的轨迹：

```text
shape=(1, 32, 3)
dtype=torch.bfloat16
device=cuda:0
```

轨迹转换后的离散动作：

```text
[1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1]
```

动作含义：

```text
1 = 向前
3 = 向右转
```

可读为：

```text
向前、向前、向前、右转、
向前、向前、向前、右转、
向前、向前、向前、向前、向前
```

这些动作只是在离线环境中生成，没有发送给真实机器人。

## 8. 显存记录

```text
模型加载峰值显存：6.69 GiB
System 2 第一轮峰值显存：7.40 GiB
System 2 第二轮峰值显存：7.61 GiB
generate_latents 峰值显存：6.37 GiB
System 1 峰值显存：5.09 GiB
```

本次运行观测到的最高显存约为：

```text
7.61 GiB
```

## 9. 关键问题与修复

最初将整个模型量化为 4-bit 时，System 1 的 `memory_encoder` 在注意力计算中报错：

```text
RuntimeError:
self and mat2 must have the same dtype,
but got BFloat16 and Byte
```

原因是标准 PyTorch `MultiheadAttention` 直接读取了量化后的 Byte 权重，与 BF16 输入类型不一致。

修复方式是：

```text
VLM 主模型继续使用 NF4 4-bit
System 1 相关模块保留 BF16
```

修改后，所有 System 1 模块均确认：

```text
参数 dtype：torch.bfloat16
是否包含 torch.uint8：False
```

## 10. 限制条件

本结果仅证明完整推理链路能够在 RTX 3080 Ti 12GB 上运行，存在以下限制：

1. 深度输入使用固定 10 米占位值，不是真实深度。
2. 仅生成 1 条候选轨迹。
3. 扩散推理仅使用 5 步，官方默认配置更高。
4. VLM 使用 NF4 4-bit，而官方配置使用 BF16。
5. 只测试了一组官方 real-world 示例。
6. 没有运行 Habitat、Isaac Sim 或真实机器人闭环控制。
7. 没有计算成功率、SPL、导航误差等论文指标。

## 11. 当前结论

已在 RTX 3080 Ti 12GB 上跑通：

- 4-bit VLM 加载；
- System 2 单图推理；
- 历史图片输入；
- 俯视图二次推理；
- 像素坐标解析；
- latent 生成；
- BF16 System 1 扩散轨迹生成；
- 轨迹到离散动作转换。

本测试属于完整双系统离线推理冒烟测试，不代表论文实验复现或真实机器人导航成功。