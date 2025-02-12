import torch
from Bio import SeqIO
import numpy as np
import pandas as pd
import csv
import os

# 加载 ESM-2 模型和字母表
model_esm2, alphabet_esm2 = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

def encode_by_esm2(seq):
    batch_converter = alphabet_esm2.get_batch_converter()
    model_esm2.eval()  # 禁用 dropout 以获得确定性结果
    
    # 准备数据
    data = [("protein", seq.upper())]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model_esm2(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    token_representations = token_representations.detach().cpu().numpy()[0][1:-1, :]

    return token_representations


def sequence_extract(fasta):
    
    target_sequence = ''
    for line in open(fasta):
        str = line.strip()
        if str[0] != '>':
            target_sequence = target_sequence + str
    return target_sequence


s=[]

with open("esm2_batch.csv","r",encoding="UTF-8-sig") as file_test:
    csv_test=csv.reader(file_test)
    for row in csv_test:
        s.append(row)
for i in range(0,1):	
    seq=sequence_extract(s[i][0])
    output=s[i][1]

    if os.path.exists(output):
        print("pass",seq)
    else:
	
        feature=encode_by_esm2(seq)
        df = pd.DataFrame(feature)
		
        df.to_csv(output, index=False, header=False)
        print("download",seq)
	
	
