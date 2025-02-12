import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
import csv
import warnings

def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn






PATH='../esm2/'
PATH1='../esm2/'





class Processor():

    def __init__(self, info_df, length):
        self.info = info_df
        self.length = length

    #ESM2
    def embedding_extract1(self, file, index1, index2):
        embed = []

        df = pd.read_csv(file, header=None)

        for i in range(index1, index2):
            if i < 0:
                embed.append([0.0] * 1280)
            else:
                try:
                    df.iloc[i, 0]
                except IndexError:
                    embed.append([0.0] * 1280)
                else:
                    embed.append(df.iloc[i].values)
        return embed



    def sequence_extract(self, file, index1, index2):
        target_sequence = ''
        sequence = ''
        for line in open(file):
            str_line = line.strip()
            if str_line[0] != '>':
                target_sequence += str_line
        for i in range(index1, index2):
            if i < 0:
                sequence += "*"
            else:
                try:
                    target_sequence[i]
                except IndexError:
                    sequence += "*"
                else:
                    sequence += target_sequence[i]
        return sequence

    def run(self):
        test_dataset = []

        for _, record in self.info.iterrows():
            uniprot_id = record['uniprot']
            resid = int(record['site']) - 1  # PTM位置


            n = int((self.length - 1) / 2)
            index1 = resid - n
            index2 = resid + n + 1

            # 提取序列
            seq_path = os.path.join(PATH, "{}.fasta".format(uniprot_id))
            seq = self.sequence_extract(seq_path, index1, index2)
            #print(seq)



            # 提取embedding1 shape(length,1280)
            embed_path1 = os.path.join(PATH1, "{}_esm2.csv".format(uniprot_id))
            embed1 = self.embedding_extract1(embed_path1, index1, index2)
            #print(embed)
            embed1 = np.array(embed1)  # 先转换为 numpy 数组
            embed1 = torch.tensor(embed1)  # 再转换为 PyTorch 张量
            #print(embed)




            m = len(seq)
            edge_index1 = dense_to_sparse(torch.ones((m, m)))[0]
            #print(edge_index1) 完全连接的图，其中每个节点与所有其他节点都有边连接


            #print("x_index shape before DataLoader:", x_index.shape)
            #print("x_index dtype before DataLoader:", x_index.dtype)
            data=Data(
                embed1=embed1.float(),
                edge_index1=edge_index1,

            )
            
            data.num_nodes=len(seq)
            #print("Data object created with label:", data.y)  # 输出 Data 对象中的标签

            group=record['set']
            if group == 'test':
                test_dataset.append(data)
            else:
                raise Exception('Unknown data group')

        return test_dataset




# 示例 DataFrame
if __name__ == '__main__':
    df = pd.read_csv('./dataset.csv')
    len_ls=[15]
    for seq_len in len_ls:
        save_path = f'./{seq_len}.pt'
        processor = Processor(df, length=seq_len)
        data_ls = processor.run()
        torch.save(data_ls,save_path)

