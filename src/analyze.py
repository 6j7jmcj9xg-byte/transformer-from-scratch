"""
消融实验分析模块
提供各种分析函数用于比较不同实验的结果
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import torch

class AblationAnalyzer:
    """消融实验分析器"""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = experiments_dir
        self.experiments_data = {}
        self.load_experiments()
    
    def load_experiments(self):
        """加载所有实验数据"""
        if not os.path.exists(self.experiments_dir):
            print(f"实验目录不存在: {self.experiments_dir}")
            return
        
        for exp_name in os.listdir(self.experiments_dir):
            exp_dir = os.path.join(self.experiments_dir, exp_name)
            summary_file = os.path.join(exp_dir, 'training_summary.json')
            config_file = os.path.join(exp_dir, 'config.json')
            
            if os.path.exists(summary_file) and os.path.exists(config_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # 加载训练日志
                log_file = os.path.join(exp_dir, 'training_log.csv')
                if os.path.exists(log_file):
                    log_data = pd.read_csv(log_file)
                else:
                    log_data = None
                
                self.experiments_data[exp_name] = {
                    'summary': summary,
                    'config': config,
                    'log': log_data,
                    'directory': exp_dir
                }
        
        print(f"已加载 {len(self.experiments_data)} 个实验")
    
    def get_comparison_table(self) -> pd.DataFrame:
        """生成实验比较表格"""
        rows = []
        for exp_name, data in self.experiments_data.items():
            summary = data['summary']
            config = data['config']
            
            row = {
                'experiment': exp_name,
                'model_type': config.get('model_type', 'complete'),
                'positional_encoding': config.get('use_positional_encoding', True),
                'embed_size': config.get('embed_size', 256),
                'num_heads': config.get('num_heads', 8),
                'num_layers': config.get('num_layers', 4),
                'best_val_loss': summary.get('best_val_loss', float('inf')),
                'final_val_loss': summary.get('final_val_loss', summary.get('best_val_loss', float('inf'))),
                'total_params': summary.get('total_parameters', 0),
                'training_time': summary.get('training_time_seconds', 0)
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_loss_comparison(self, save_path: str = None):
        """绘制损失比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        df = self.get_comparison_table()
        
        # 1. 模型类型比较
        if 'model_type' in df.columns:
            model_loss = df.groupby('model_type')['best_val_loss'].mean().sort_values()
            axes[0, 0].bar(model_loss.index, model_loss.values)
            axes[0, 0].set_title('Best Validation Loss by Model Type')
            axes[0, 0].set_ylabel('Validation Loss')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 参数量与性能关系
        axes[0, 1].scatter(df['total_params'], df['best_val_loss'], alpha=0.7)
        axes[0, 1].set_xlabel('Total Parameters')
        axes[0, 1].set_ylabel('Best Validation Loss')
        axes[0, 1].set_title('Model Size vs Performance')
        axes[0, 1].set_xscale('log')
        
        # 3. 位置编码效果
        if 'positional_encoding' in df.columns:
            pos_encoding_loss = df.groupby('positional_encoding')['best_val_loss'].mean()
            axes[1, 0].bar(['No PE', 'With PE'], pos_encoding_loss.values)
            axes[1, 0].set_title('Effect of Positional Encoding')
            axes[1, 0].set_ylabel('Validation Loss')
        
        # 4. 训练时间比较
        axes[1, 1].bar(df['experiment'], df['training_time'] / 60)  # 转换为分钟
        axes[1, 1].set_title('Training Time by Experiment')
        axes[1, 1].set_ylabel('Training Time (minutes)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, experiments: List[str] = None, save_path: str = None):
        """绘制训练曲线比较"""
        if experiments is None:
            experiments = list(self.experiments_data.keys())
        
        plt.figure(figsize=(12, 8))
        
        for exp_name in experiments:
            if exp_name in self.experiments_data and self.experiments_data[exp_name]['log'] is not None:
                log_data = self.experiments_data[exp_name]['log']
                if 'train_loss' in log_data.columns and 'val_loss' in log_data.columns:
                    plt.plot(log_data['epoch'], log_data['val_loss'], 
                            label=f"{exp_name} (val)", linewidth=2)
                    plt.plot(log_data['epoch'], log_data['train_loss'], 
                            label=f"{exp_name} (train)", linestyle='--', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_text_comparison(self, start_sequence: str = "Shall I compare thee", max_length: int = 200):
        """生成文本比较"""
        from model import create_model
        from data import TinyShakespeareDataset
        
        dataset = TinyShakespeareDataset()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("文本生成比较:")
        print("=" * 60)
        
        for exp_name, data in self.experiments_data.items():
            model_path = os.path.join(data['directory'], 'final_model.pth')
            
            if os.path.exists(model_path):
                # 重新创建模型
                config = data['config']
                model = create_model(len(dataset.chars), config)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                
                # 生成文本
                generated_text = self._generate_text_single(model, start_sequence, device, dataset, max_length)
                
                print(f"\n{exp_name}:")
                print(f"配置: embed_size={config['embed_size']}, heads={config['num_heads']}, layers={config['num_layers']}")
                print(f"最佳验证损失: {data['summary']['best_val_loss']:.4f}")
                print(f"生成文本: {generated_text}")
                print("-" * 60)
    
    def _generate_text_single(self, model, start_sequence, device, dataset, max_length):
        """为单个模型生成文本"""
        model.eval()
        input_seq = torch.tensor([dataset.char_to_idx.get(char, 0) for char in start_sequence]).unsqueeze(0).to(device)
        
        generated_text = start_sequence
        
        for _ in range(max_length - len(start_sequence)):
            with torch.no_grad():
                seq_len = input_seq.shape[1]
                mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)
                
                output = model(input_seq, trg_mask=mask)
                predicted_idx = torch.argmax(output[:, -1, :], dim=-1).item()
                predicted_char = dataset.idx_to_char[predicted_idx]
                generated_text += predicted_char
                
                new_token = torch.tensor([[predicted_idx]]).to(device)
                input_seq = torch.cat((input_seq, new_token), dim=1)
                
                if input_seq.shape[1] > model.seq_length:
                    input_seq = input_seq[:, -model.seq_length:]
        
        return generated_text
    
    def create_ablation_report(self, save_path: str = "ablation_report.md"):
        """生成消融实验报告"""
        df = self.get_comparison_table()
        
        report = [
            "# Transformer 消融实验报告",
            f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"实验数量: {len(df)}",
            "\n## 实验概览\n"
        ]
        
        # 实验概览表格
        report.append(df.to_markdown(index=False))
        
        # 关键发现
        report.extend([
            "\n## 关键发现\n",
            f"- **最佳模型**: {df.loc[df['best_val_loss'].idxmin(), 'experiment']} (损失: {df['best_val_loss'].min():.4f})",
            f"- **最快训练**: {df.loc[df['training_time'].idxmin(), 'experiment']} ({df['training_time'].min()/60:.1f} 分钟)",
            f"- **最小模型**: {df.loc[df['total_params'].idxmin(), 'experiment']} ({df['total_params'].min():,} 参数)",
            f"- **最大模型**: {df.loc[df['total_params'].idxmax(), 'experiment']} ({df['total_params'].max():,} 参数)"
        ])
        
        # 写入报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"消融实验报告已保存至: {save_path}")

def main():
    """主函数：运行完整分析"""
    analyzer = AblationAnalyzer()
    
    # 生成比较表格
    df = analyzer.get_comparison_table()
    print("实验比较表格:")
    print(df.to_string(index=False))
    
    # 绘制图表
    analyzer.plot_loss_comparison("ablation_comparison.png")
    analyzer.plot_training_curves(save_path="training_curves.png")
    
    # 生成文本比较
    analyzer.generate_text_comparison()
    
    # 生成报告
    analyzer.create_ablation_report()

if __name__ == "__main__":
    main()