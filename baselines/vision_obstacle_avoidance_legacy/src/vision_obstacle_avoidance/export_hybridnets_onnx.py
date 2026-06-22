#!/usr/bin/env python3
"""
HybridNets PyTorch → ONNX 导出脚本。
将训练好的 HybridNets 模型转换为 ONNX 格式，供 HybridNetsEngine 加载。

使用方法:
  1. 准备 PyTorch 权重文件 (.pth)
  2. python export_hybridnets_onnx.py --weights path/to/model.pth --output models/hybridnets.onnx

来源: https://github.com/datvuthanh/HybridNets
"""

import argparse
import sys
import os


def export_onnx(weights_path: str, output_path: str,
                input_width: int = 640, input_height: int = 384):
    """
    导出 HybridNets 模型到 ONNX 格式。

    参数:
      weights_path: PyTorch .pth 权重文件路径
      output_path: ONNX 输出路径
      input_width, input_height: 模型输入尺寸
    """
    try:
        import torch
        import onnx
        import onnxruntime
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}")
        print("请运行: pip install torch onnx onnxruntime")
        return False

    # ── 1. 加载模型 ──
    print(f"[1/4] 加载模型权重: {weights_path}")
    try:
        # HybridNets 模型定义（需与训练时一致）
        from hybridnets_engine import HybridNetsEngine
        # HybridNetsEngine 仅用于 ONNX 推理，导出需要完整模型定义
        # 此处需要从 HybridNets 源码导入模型
        print("[警告] 需要 HybridNets 源码库。请 clone 仓库后重试：")
        print("  git clone https://github.com/datvuthanh/HybridNets")
        print("  pip install -e HybridNets/")
        print()
        print("  然后修改本脚本中的模型导入路径。")
    except ImportError:
        pass

    # ── 尝试方式 1: 从本地 HybridNets 克隆加载 ──
    try:
        sys.path.insert(0, os.path.expanduser("~/HybridNets"))
        from model import HybridNets
        from utils.utils import BBoxTransform, ClipBoxes

        model = HybridNets(compound_coef=3, num_classes=4)
        checkpoint = torch.load(weights_path, map_location='cpu',
                               weights_only=True)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("  模型加载成功 (HybridNets 源码)")
    except Exception as e:
        print(f"  方式1失败: {e}")
        print()
        print("[替代方案] 如果您没有完整的 HybridNets 训练权重，可以使用以下方式：")
        print()
        print("  A. 下载预训练权重:")
        print("     wget https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth")
        print()
        print("  B. 或直接使用 HBD-Net-RT 作为主 DL 引擎 (已集成，无需 ONNX)")
        print("     HBD-Net-RT: RepVGG-lite + 5 Head, PyTorch 原生推理，GPU 6ms/帧")
        print()
        return False

    # ── 2. 导出 ONNX ──
    print(f"[2/4] 导出 ONNX (输入: {input_height}x{input_width})...")
    dummy_input = torch.randn(1, 3, input_height, input_width)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['regression', 'classification', 'segmentation'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'regression': {0: 'batch_size'},
            'classification': {0: 'batch_size'},
            'segmentation': {0: 'batch_size'},
        },
    )
    print(f"  ONNX 已保存: {output_path}")

    # ── 3. 验证 ──
    print(f"[3/4] 验证 ONNX 模型...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX 模型结构验证通过")

    # ── 4. 推理测试 ──
    print(f"[4/4] 推理测试...")
    sess = onnxruntime.InferenceSession(
        output_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    test_input = dummy_input.numpy()
    outputs = sess.run(None, {input_name: test_input})
    print(f"  推理成功: {len(outputs)} 个输出")
    for i, o in enumerate(outputs):
        print(f"    输出[{i}]: shape={o.shape}")

    print()
    print("=" * 60)
    print(f"✅ 导出完成: {output_path}")
    print(f"   文件大小: {os.path.getsize(output_path) / 1e6:.1f} MB")
    print("=" * 60)

    # 更新 config
    print()
    print(f"[提示] 在 config.py 中设置:")
    print(f"  HYBRIDNETS_ONNX_PATH = '{output_path}'")
    print(f"  HYBRIDNETS_USE_GPU = True  # 如需 GPU 推理")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="HybridNets PyTorch → ONNX 导出")
    parser.add_argument("--weights", "-w", required=True,
                       help="PyTorch 权重文件路径 (.pth)")
    parser.add_argument("--output", "-o", default="models/hybridnets.onnx",
                       help="ONNX 输出路径 (默认: models/hybridnets.onnx)")
    parser.add_argument("--width", type=int, default=640,
                       help="输入宽度 (默认: 640)")
    parser.add_argument("--height", type=int, default=384,
                       help="输入高度 (默认: 384)")
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    success = export_onnx(args.weights, args.output, args.width, args.height)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
