import os
import torch
import torch.optim as optim
import torch.nn as nn
from model import CompleteTransformer
from data import TinyShakespeareDataset
from utils import save_model, plot_loss
import argparse
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import json

# ------------------------ 设置随机种子 ------------------------

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------ 解析命令行参数 ------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Model for TinyShakespeare")

    parser.add_argument('--model_type', type=str, default='complete', choices=['complete'], help="Type of transformer model")
    parser.add_argument('--use_positional_encoding', type=str, default='True', help="Whether to use positional encoding")
    parser.add_argument('--embed_size', type=int, default=256, help="Embedding size")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads")
    parser.add_argument('--num_layers', type=int, default=4, help="Number of transformer layers")
    parser.add_argument('--epochs', type=int, default=50, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument('--experiment_name', type=str, default=None, help="Experiment name for logging")
    parser.add_argument('--save_dir', type=str, default='experiments', help="Directory to save experiment results")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")  # 添加 seed 参数
    parser.add_argument('--seq_length', type=int, default=128, help="Sequence length for training")  # 添加 seq_length 参数

    return parser.parse_args()

# ------------------------ 创建实验目录 ------------------------

def create_experiment_dir(args):
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.model_type}_e{args.embed_size}_h{args.num_heads}_l{args.num_layers}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    experiment_dir = os.path.join(args.save_dir, experiment_name)
    # 创建实验目录
    os.makedirs(experiment_dir, exist_ok=True)

    # 保存实验配置
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    return experiment_dir, experiment_name

# ------------------------ 训练函数 ------------------------

def train():
    args = parse_args()
    set_random_seed(args.seed)

    # 创建实验目录
    experiment_dir, experiment_name = create_experiment_dir(args)
    print(f"Experiment: {experiment_name}")
    print(f"Results will be saved to: {experiment_dir}")

    # 数据集初始化
    dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dataset = TinyShakespeareDataset(url=dataset_url, seq_length=args.seq_length)

    # 分割训练集和验证集
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 模型初始化
    model = CompleteTransformer(
        vocab_size=len(dataset.chars), 
        embed_size=args.embed_size, 
        num_heads=args.num_heads, 
        num_layers=args.num_layers, 
        seq_length=args.seq_length,
        use_positional_encoding=True,
        use_decoder=False,  # 只使用编码器部分
        ff_expansion=4,
        dropout=0.1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 保存损失数据
    train_losses = []
    val_losses = []

    # 创建保存损失曲线图的目录
    runs_dir = "results/runs"
    os.makedirs(runs_dir, exist_ok=True)

    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (input_seq, target_seq) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")):
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)

            # 直接输入序列到模型中，使用编码器部分
            output = model(input_seq, mask=None)
            loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

        # 保存训练损失
        train_losses.append(avg_train_loss)

        # 计算验证损失
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_seq, target_seq in val_loader:
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)
                output = model(input_seq, mask=None)
                loss = criterion(output.view(-1, output.size(-1)), target_seq.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # 保存验证损失
        val_losses.append(avg_val_loss)

        # 保存模型
        save_model(model, os.path.join(experiment_dir, 'final_model.pth'))

        # 绘制损失曲线并保存到 `runs/` 目录
        plot_loss(train_losses, val_losses, output_dir=runs_dir)

if __name__ == "__main__":
    train()
