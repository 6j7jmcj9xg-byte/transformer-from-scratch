#!/usr/bin/env python3
"""
运行消融实验的脚本
批量运行不同配置的实验
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# 实验配置列表
ABLATION_CONFIGS = [
    # 基准配置
    {
        "name": "baseline",
        "model_type": "complete",
        "use_positional_encoding": True,
        "embed_size": 256,
        "num_heads": 8,
        "num_layers": 4,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    # 无位置编码
    {
        "name": "no_positional_encoding",
        "model_type": "no_pos", 
        "use_positional_encoding": False,  # 明确设置为 False
        "embed_size": 256,
        "num_heads": 8,
        "num_layers": 4,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    # 小型模型
    {
        "name": "small_model",
        "model_type": "small",
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 128,
        "num_heads": 4,
        "num_layers": 2,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    # 大型模型
    {
        "name": "large_model", 
        "model_type": "large",
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 512,
        "num_heads": 16,
        "num_layers": 8,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-4
    },
    # 可学习位置编码
    {
        "name": "learned_positional_encoding",
        "model_type": "learned_pe",
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 256,
        "num_heads": 8, 
        "num_layers": 4,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    # 不同头数
    {
        "name": "4_heads",
        "model_type": "complete",
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 256,
        "num_heads": 4,
        "num_layers": 4,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    {
        "name": "16_heads",
        "model_type": "complete", 
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 256,
        "num_heads": 16,
        "num_layers": 4,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    # 不同层数
    {
        "name": "2_layers",
        "model_type": "complete",
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 256,
        "num_heads": 8,
        "num_layers": 2,
        "epochs": 50, 
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    {
        "name": "8_layers",
        "model_type": "complete",
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 256,
        "num_heads": 8,
        "num_layers": 8,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    # 不同嵌入维度
    {
        "name": "embed_128",
        "model_type": "complete",
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 128,
        "num_heads": 8,
        "num_layers": 4,
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4
    },
    {
        "name": "embed_512", 
        "model_type": "complete",
        "use_positional_encoding": True,  # 添加这一行
        "embed_size": 512,
        "num_heads": 8,
        "num_layers": 4,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-4
    }
]

def run_experiment(config, save_dir="experiments"):
    """运行单个实验"""
    cmd = [
        "python", "src/train.py",
        "--model_type", config["model_type"],
        "--use_positional_encoding", str(config["use_positional_encoding"]),
        "--embed_size", str(config["embed_size"]),
        "--num_heads", str(config["num_heads"]),
        "--num_layers", str(config["num_layers"]),
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--experiment_name", config["name"],
        "--save_dir", save_dir
    ]
    
    print(f"运行实验: {config['name']}")
    print("命令:", " ".join(cmd))
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, check=True)
        if result.returncode == 0:
            print(f"✓ 实验 {config['name']} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 实验 {config['name']} 失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument("--configs", nargs="+", help="运行特定配置的实验")
    parser.add_argument("--save_dir", default="experiments", help="保存实验结果的目录")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在的实验")
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 选择要运行的配置
    if args.configs:
        configs_to_run = [cfg for cfg in ABLATION_CONFIGS if cfg["name"] in args.configs]
    else:
        configs_to_run = ABLATION_CONFIGS
    
    print(f"准备运行 {len(configs_to_run)} 个实验")
    
    # 运行实验
    successful = 0
    for config in configs_to_run:
        # 检查是否已存在
        exp_dir = os.path.join(args.save_dir, config["name"])
        if args.skip_existing and os.path.exists(exp_dir):
            print(f"跳过已存在的实验: {config['name']}")
            continue
            
        if run_experiment(config, args.save_dir):
            successful += 1
    
    print(f"\n实验完成: {successful}/{len(configs_to_run)} 成功")
    
    # 运行分析
    if successful > 0:
        print("\n运行分析...")
        subprocess.run(["python", "src/analyze.py"])

if __name__ == "__main__":
    main()