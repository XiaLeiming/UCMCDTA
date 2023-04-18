# UCMCDTA

This repo contains the code for our paper " Drug-target binding affinity prediction using message passing neural network and self supervised method" 

by Leiming Xia, Zhen Li*

We report UCMCDTA, an deep learning model for binding affinity prediction tasks. This model  using an undirected-CMPNN for molecule embedding and MCPCProt models for protein embedding. Both embeddings are concatenated for DTA prediction. The results showed that the proposed model outperformed other deep learning methods, which also provides a novel strategy for deep learning-based virtual screening methods.  

# Dependencies

* Python 3.7
* Pytorch
* numpy
* pickle
* RDKit
* sklearn
* CUDA

# Usage

to test BindingDB Ki dataset: 
first

```
cd ki
```

and then run

```
python dataProcess.py 
```

to generate the files required. Then

```
cd ..
python main.py --dataset ki --mode regression
```

# Acknowledgement
We'd like to express our gratitude towards all the colleagues and reviewers for helping us improve the paper. 
