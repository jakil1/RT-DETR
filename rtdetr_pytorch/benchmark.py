import time
import torch
import argparse
import torch.nn as nn
from thop import profile, clever_format
from src.core import YAMLConfig


def main(args):
    # 1. 加载配置和模型
    print(f"正在加载配置文件: {args.config} ...")
    cfg = YAMLConfig(args.config, resume=args.resume)

    # 构建模型
    model = cfg.model

    # 准备设备
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    # 定义输入大小 (RT-DETR 默认为 640x640)
    input_shape = (1, 3, 640, 640)
    dummy_input = torch.randn(input_shape).to(device)

    print("------------------------------------------------")
    print(f"模型: RT-DETR (Backbone: {args.config.split('/')[-1]})")
    print(f"输入尺寸: {input_shape}")
    print("------------------------------------------------")

    # -------------------------------------------------------
    # 2. 计算 Params 和 GFLOPs
    # -------------------------------------------------------
    print("正在计算 Params 和 GFLOPs ...")
    try:
        # 自定义操作处理，防止部分算子报错
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
        print(f"👉 Params (参数量): {params_fmt}")
        print(f"👉 FLOPs (计算量):  {macs_fmt}")
        print("(注意: 1 GFLOPs ≈ 2 * MACs，通常论文汇报 GFLOPs)")
    except Exception as e:
        print(f"计算 FLOPs 失败 (可能是算子不支持): {e}")

    # -------------------------------------------------------
    # 3. 计算 FPS (推理速度)
    # -------------------------------------------------------
    print("------------------------------------------------")
    print("正在测试 FPS (预热 50 次，循环 200 次) ...")

    # 预热 (Warm up) - 让 GPU 进入工作状态
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    # 正式计时
    torch.cuda.synchronize()
    start_time = time.time()

    t_steps = 200
    with torch.no_grad():
        for _ in range(t_steps):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / t_steps
    fps = 1.0 / avg_time

    print(f"👉 平均推理时间: {avg_time * 1000:.2f} ms")
    print(f"👉 FPS (帧率):    {fps:.2f}")
    print("------------------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('-r', '--resume', type=str, default=None, help='权重文件路径 (可选)')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='使用设备')
    args = parser.parse_args()
    main(args)