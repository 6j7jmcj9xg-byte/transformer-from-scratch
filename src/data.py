import torch
import numpy as np
import requests

class TinyShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, url, seq_length=128):
        """
        初始化数据集，加载从URL下载的文本文件并进行处理
        :param url: 数据集的URL
        :param seq_length: 每个样本的序列长度
        """
        # 下载数据集
        response = requests.get(url)
        if response.status_code == 200:
            self.text = response.text
        else:
            raise Exception(f"Failed to download the data from {url}")

        # 构建字符到整数的映射
        self.chars = sorted(set(self.text))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        # 将文本转化为索引
        self.text_as_int = np.array([self.char_to_idx[char] for char in self.text])
        
        # 定义序列长度
        self.seq_length = seq_length

    def __len__(self):
        """返回数据集的大小"""
        return len(self.text_as_int) // self.seq_length

    def __getitem__(self, idx):
        """获取一个样本，返回输入序列和目标序列"""
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length + 1
        seq = self.text_as_int[start_idx:end_idx]
        
        input_seq = torch.tensor(seq[:-1])  # 输入序列，去掉最后一个字符
        target_seq = torch.tensor(seq[1:])  # 目标序列，去掉第一个字符
        return input_seq, target_seq
