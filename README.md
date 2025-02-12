# PhosLLPS
PhosLLPS, a Transformer-based GNN prediction medthod (AUC=0.9116) to identify functional phosphorylation sites that regulate liquid-liquid phase separation (LLPS)

![Logo](/model.png)
  * The complete protein sequence (with a length of L amino acids) is input into the pre-trained protein language model ESM2 to obtain L x 1280-dimensional features. Then, a 15 x 1280-dimensional feature set is derived as node features (where 15 represents the 7 amino acids upstream and downstream of the phosphorylation sites). A 15 x 15 full adjacency matrix is used as the graph's edge features. Next, the graph is processed through three TransformerConv layers, each employing a self-attention mechanism. Finally, the processed features are input into a multi-layer perceptron (MLP) to generate the final prediction results.

# Contact
<1>haifengchen@sjtu.edu.cn
<2>hongxk@fzu.edu.cn

## Set up environment
1. Copy Python library
  * git clone https://github.com/HFChenLab/PhosLLPS.git
  * cd PhosLLPS

2. Set up a Conda environment
  * conda env create -f environment.yml
  * conda activate PLM2

3. install esm software
  * pip install fair-esm

## Obtain embedding features of pretrained protein language model ESM2
  * cd esm2
  * prepare P00533.fasta 
  * prepare esm2_batch.csv
  * python esm2.py


## Extract key features from above embedding features
  * cd dataset
  * prepare dataset.csv, including uniprot and site
  * python generate_dataset.py 

## Predict prob and pred of each site of proteins
  * cd predict
  * python predict.py
