<<<<<<< HEAD
# Tiny Shakespeare Transformer 模型

该项目实现了一个基于 Transformer 的字符级语言建模模型，使用 `Tiny Shakespeare` 数据集进行训练。项目包括训练脚本、消融实验脚本、结果分析以及文本生成等功能。

## 项目结构

project/
├── src/
│ ├── init.py
│ ├── data.py
│ ├── analyze.py
│ ├── model.py
│ ├── utils.py
│ └── train.py
├── requirements.txt
├── README.md
├── scripts/
│ └── run.sh
│ └── run_ablation.sh
│ └── run_analysis.sh
└── results/
├── runs
├── experiments


### `src/` 文件夹
包含所有源代码文件，功能如下：
- **`data.py`**：负责加载和处理数据，将 `Tiny Shakespeare` 数据集转换成模型可以使用的格式，包含 `TinyShakespeareDataset` 类。
- **`model.py`**：定义了 Transformer 模型的架构，包含编码器（Encoder）和解码器（Decoder）部分。核心组件包括 `MultiHeadAttention`、`PositionalEncoding`、`Encoder`、`Decoder` 和 `CompleteTransformer`。
- **`train.py`**：模型训练脚本，负责训练过程中的损失计算、优化、保存模型等操作。
- **`utils.py`**：包含一些常用工具函数，如保存和加载模型、绘制损失曲线、调整学习率等。
- **`analyze.py`**：提供消融实验分析工具 `AblationAnalyzer`，用于分析不同实验配置下模型的性能，并生成相应的图表。

### `scripts/` 文件夹
包含批处理脚本，用于运行实验和分析：
- **`run_ablation.sh`**：运行多个消融实验，批量执行不同配置的实验。
- **`run_analysis.sh`**：运行分析脚本，比较不同实验的结果，生成损失图、训练曲线等。
- **`run.sh`**：运行单个实验的脚本，调用 `src/train.py` 进行模型训练。

### `results/` 文件夹
用于存储实验结果：
- **`runs/`**：存储每个实验的结果，包括模型的训练和验证损失曲线等。
- **`experiments/`**：存储所有实验的输出，每个实验有一个子目录，保存该实验的训练结果和配置文件。

## 安装依赖

首先，确保你的环境中已经安装了 `Python` 和 `pip`，然后使用以下命令安装项目所需的依赖：

```bash
pip install -r requirements.txt

使用方法
训练模型

你可以通过以下命令训练模型：
python3 src/train.py --model_type complete --embed_size 256 --num_heads 8 --num_layers 4 --epochs 50 --batch_size 64 --learning_rate 0.0001

运行消融实验
消融实验通过以下命令批量运行：
python3 scripts/run_ablation.sh

分析实验结果
分析实验结果可以通过以下命令执行：
python3 scripts/run_analysis.sh

消融实验
本项目支持多种消融实验，针对 Transformer 模型的不同配置（如注意力头数、层数、嵌入维度等）进行测试。每次实验的结果将被保存并用于后续分析，比较不同配置的性能。

结果展示
每次训练过程中，模型的训练和验证损失将通过损失曲线图展示，并保存到 results/runs 文件夹。你可以通过这些图表了解模型训练的进展和效果。

此项目展示了 Transformer 模型在字符级语言建模中的应用，并提供了便于实验、分析和调试的工具。

### 说明：

1. **项目结构**：在 `README.md` 中，列出了项目的文件夹结构并对每个文件夹和文件的功能进行了详细说明。
2. **使用方法**：描述了如何安装依赖、训练模型、运行消融实验和分析实验结果的命令。
3. **消融实验与结果展示**：介绍了消融实验的功能，并说明了如何展示实验结果。
=======
# transformer-from-scratch
大模型期中作业
>>>>>>> da700cc35de334faefeecc39a65b6d5a447d1def
