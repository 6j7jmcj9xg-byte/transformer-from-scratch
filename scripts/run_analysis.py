#!/usr/bin/env python3
"""
运行分析脚本 - 修正版本
"""

import sys
import os

# 添加项目根目录到Python路径，这样可以从src导入模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from src.analyze import AblationAnalyzer
    print("成功导入分析模块")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保:")
    print("1. 项目结构正确，src目录包含analyze.py")
    print("2. analyze.py中包含AblationAnalyzer类")
    sys.exit(1)

def main():
    # 初始化分析器
    analyzer = AblationAnalyzer("experiments")
    
    # 检查是否有实验数据
    if not analyzer.experiments_data:
        print("未找到实验数据！")
        print("请先运行消融实验，确保experiments目录包含实验结果")
        return
    
    print(f"找到 {len(analyzer.experiments_data)} 个实验数据")
    
    # 生成比较表格
    print("\n" + "="*50)
    print("实验比较表格")
    print("="*50)
    df = analyzer.get_comparison_table()
    print(df.to_string(index=False))
    
    # 保存比较表格到CSV
    df.to_csv("experiment_comparison.csv", index=False)
    print("\n比较表格已保存至: experiment_comparison.csv")
    
    # 绘制损失比较图
    print("\n生成损失比较图...")
    analyzer.plot_loss_comparison("ablation_comparison.png")
    
    # 绘制训练曲线
    print("生成训练曲线图...")
    analyzer.plot_training_curves(save_path="training_curves.png")
    
    # 文本生成比较
    print("\n" + "="*50)
    print("文本生成比较")
    print("="*50)
    analyzer.generate_text_comparison()
    
    # 生成完整报告
    print("\n生成消融实验报告...")
    analyzer.create_ablation_report()
    
    print("\n" + "="*50)
    print("分析完成！")
    print("生成的文件:")
    print("- ablation_comparison.png (损失比较图)")
    print("- training_curves.png (训练曲线图)") 
    print("- ablation_report.md (完整报告)")
    print("- experiment_comparison.csv (实验比较表格)")
    print("="*50)

if __name__ == "__main__":
    main()