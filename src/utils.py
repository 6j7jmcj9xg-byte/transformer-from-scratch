import torch
import os
import matplotlib.pyplot as plt

def save_model(model, path):
    """
    保存模型参数
    :param model: 训练好的模型
    :param path: 保存路径
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    加载模型参数
    :param model: 未训练的模型
    :param path: 模型参数文件路径
    :return: 加载了参数的模型
    """
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def plot_loss(train_losses, val_losses, output_dir="results"):
    """
    绘制训练和验证损失曲线并保存
    :param train_losses: 训练损失列表
    :param val_losses: 验证损失列表
    :param output_dir: 输出结果的保存目录
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "train_curve.png"))
    plt.close()

def get_lr_scheduler(optimizer, step_size=10, gamma=0.8):
    """
    获取学习率调度器
    :param optimizer: 优化器
    :param step_size: 每隔多少步更新一次学习率
    :param gamma: 学习率衰减因子
    :return: 学习率调度器
    """
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def log_train_metrics(epoch, train_loss, val_loss, logfile="train_log.txt"):
    """
    记录每个epoch的训练和验证损失
    :param epoch: 当前epoch
    :param train_loss: 训练损失
    :param val_loss: 验证损失
    :param logfile: 日志文件路径
    """
    if val_loss is None:
        val_loss = 0.0

    with open(logfile, "a") as log:
        log.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")
    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

def clip_gradients(model, max_norm=1.0):
    """
    对梯度进行裁剪，防止梯度爆炸
    :param model: 需要裁剪梯度的模型
    :param max_norm: 梯度裁剪的阈值
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
