# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:18:27 2020

@author: shuyu
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# cmpnn
from chemprop.models.mpn import MPN


class GanDTI(nn.Module):
    def __init__(self, compound_len, protein_len, features, GNN_depth, MLP_depth, mode, args):  # 多了一个args
        super(GanDTI, self).__init__()
        self.mode = mode
        self.embed_compound = nn.Embedding(compound_len, features)
        self.embed_protein = nn.Embedding(protein_len, features)
        self.GNN_depth = GNN_depth
        self.GNN = nn.ModuleList(nn.Linear(features, features) for i in range(GNN_depth))
        self.W_att = nn.Linear(features, features)
        self.MLP_depth = MLP_depth
        self.MLP = nn.ModuleList(nn.Linear(features*2, features*2) for i in range(self.MLP_depth))
        self.classification_out = nn.Linear(2*features, 2)
        self.regression_out = nn.Linear(2*features, 1)
        self.protein_512_40 = nn.Linear(512,40)
        self.dropout = nn.Dropout(0.5)
        self.args = args
#cmpnn
        self.cmpnn_encoder = MPN(args)
        self.device = torch.device('cuda')
        self.L1ceng30_1 = nn.Linear(30,1)
        self.L2ceng1001_256 = nn.Linear(1001,int(512/2))
        self.L3ceng512_256 = nn.Linear(512, int(512/2))

    def Attention(self, compound, protein):
        compound_h = torch.relu(self.W_att(compound))
        protein_h = torch.relu(self.W_att(protein))
        mult = compound @ protein_h.T
        weights = torch.tanh(mult)
        protein = weights.T * protein_h
        protein_vector = torch.unsqueeze(torch.mean(protein, 0), 0)
        return protein_vector
        
    def GraphNeuralNet(self, compound, A, GNN_depth):
        residual = compound
        for i in range(GNN_depth):
            compound_h = F.leaky_relu(self.GNN[i](compound))
            compound = compound + torch.matmul(A,compound_h)
        compound = compound + residual
        compound_vector = torch.unsqueeze(torch.mean(compound, 0), 0)
        return compound_vector
    
    def MLP_module(self, compound_protein, MLP_depth, mode):
        for i in range(MLP_depth):
            compound_protein = torch.relu(self.MLP[i](compound_protein))
        compound_protein = self.dropout(compound_protein)
        if mode == 'classification':
            out = self.classification_out(compound_protein)
        elif mode == 'regression':
            out = self.regression_out(compound_protein)
        return out
    
    def forward(self, data):
        smiless_1_batch, A, protein1 = data
        features_batch = None
        compound_vector = self.cmpnn_encoder(smiless_1_batch, features_batch)
        proteins_tensor = []
        for i in protein1[:-1]:
            proteins_tensor.append(torch.from_numpy(np.array(i)).float().to(self.device))
        proteins_tensor= torch.tensor(proteins_tensor).to(self.device)
        A69_protein_tensor = protein1[-1].to(self.device)
        A69_protein_tensor = self.L1ceng30_1(A69_protein_tensor)
        A69_protein_tensor = A69_protein_tensor.reshape([1001])
        A69_256_protein_tensor = self.L2ceng1001_256(A69_protein_tensor)
        pp = proteins_tensor.shape[0]
        proteins_tensor = self.L3ceng512_256(proteins_tensor)
        A69_256_protein_tensor = A69_256_protein_tensor.reshape([1,256])
        proteins_tensor = proteins_tensor.reshape([1,256])
        proteins_tensor = torch.cat((A69_256_protein_tensor, proteins_tensor), 1)
        protein_512_40vector = self.protein_512_40(proteins_tensor.float())
        protein_vector = self.Attention(compound_vector, protein_512_40vector)
        compound_protein = torch.cat((compound_vector, protein_vector), 1)
        out = self.MLP_module(compound_protein, self.MLP_depth, self.mode)
        return out
    
    def __call__(self, data, train=True):
        feature_data, label_data = data[:-1], data[-1]
        predict_data = self.forward(feature_data)
        if train:
            if self.mode == 'classification':
                output = F.cross_entropy(predict_data, label_data)
            elif self.mode == 'regression':
                loss = nn.MSELoss()
                output = loss(predict_data[0].float(), label_data.float())
            return output
        else:
            labels = label_data.to('cpu').data.numpy()
            if self.mode == 'classification':
                predict_data = torch.sigmoid(predict_data).to('cpu').data.numpy()
                predict_result = list(map(lambda x: np.argmax(x), predict_data))
                predict_score = list(map(lambda x: x[1], predict_data))
            elif self.mode == 'regression':
                predict_result = predict_data[0].to('cpu').data.numpy()
                predict_score = predict_result
            return labels, predict_result, predict_score
