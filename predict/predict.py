import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset


from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import TransformerConv

import numpy as np
import pandas as pd


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, matthews_corrcoef,precision_recall_curve,roc_curve,auc

import time

import math

torch.manual_seed(42)          # 设置CPU随机种子
torch.cuda.manual_seed(42)     # 设置GPU随机种子（如果使用GPU）
torch.cuda.manual_seed_all(42) # 如果使用多个GPU
torch.backends.cudnn.deterministic = True  # 保证每次的结果一致
torch.backends.cudnn.benchmark = False     # 确保可复现性（但可能影响性能）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print("-------预测开始-------")


global_window_size = 15
global_input_dim = 1280


class GNNTrans(nn.Module):
    def __init__(self, input_dim=global_input_dim, hidden_dim=64, num_layers=3, noise_std=0.12):
        super(GNNTrans, self).__init__()
        self.input_dim = input_dim
        self.noise_std = noise_std
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define convolutional layers
        self.convs = torch.nn.ModuleList(
            [TransformerConv(in_channels=input_dim, out_channels=hidden_dim, heads=1)] +
            [TransformerConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=1)
             for _ in range(num_layers - 1)]
        )

        # BatchNorm layers
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # MLP layer for final output
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # BatchNorm 在 Linear 后面
            nn.ReLU(),
            
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),  # BatchNorm 在 Linear 后面
            nn.ReLU(),
            
            nn.Dropout(p=0.5),
            nn.Linear(64, 2),
        )

    def add_gaussian_noise(self, x):
        """
        在训练时为输入添加高斯噪声
        """
        if self.training and self.noise_std > 0:
            noise = torch.normal(mean=0, std=self.noise_std, size=x.size(), device=x.device)
            x = x + noise
        return x


    def get_conv_result(self, x, edge_index):
        """
        经过图卷积层处理数据
        """
        for i in range(self.num_layers):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        return x

    def forward(self, data):
        # 拼接输入特征
        #x = torch.cat((data.embed1, data.embed5), dim=-1)
        x = data.embed1

        x = self.add_gaussian_noise(x)

        edge_index, batch = data.edge_index1, data.batch
        idx = (data.ptr + int(global_window_size / 2))[:-1]

        # 图卷积操作
        x = self.get_conv_result(x, edge_index)

        # 根据索引选择相关部分
        x = x[idx]

        # 添加高斯噪声后经过 MLP
        #x = self.add_gaussian_noise(x)
        out = self.mlp(x)

        return out



# 加载数据集
val_dataset = torch.load('../dataset/15.pt',weights_only=False)
batch_size_value = 128

val_dataloader = DataLoader(val_dataset, batch_size=batch_size_value, shuffle=False)


# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GNNTrans().to(device)
model.load_state_dict(torch.load('model_weights.pth', map_location=device, weights_only=True))

model.eval()

# 进行预测
all_preds = []
all_probs = []
start_time=time.time()

total_batches = len(val_dataloader)  # 总批次数

with torch.no_grad():
    for i, batch in enumerate(val_dataloader, 1):  # 1-based index
        batch = batch.to(device)

        outputs = model(batch)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 获取类 1 概率
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # 获取预测类别（0 或 1）

        all_preds.extend(preds)
        all_probs.extend(probs)

        batch_time = time.time() - start_time

        # 打印进度
        print(f"Batch {i}/{total_batches} processed in {batch_time:.2f} seconds", end="\r")

print("\n预测完成，正在保存结果...")

# 保存结果
df = pd.DataFrame({'all_preds': all_preds, 'all_probs': all_probs})
df.to_csv('predictions.csv', index=False)

print("预测完成，结果已保存至 predictions.csv")




