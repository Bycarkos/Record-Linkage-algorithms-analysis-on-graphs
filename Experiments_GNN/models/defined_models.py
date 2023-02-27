from hydra.utils import instantiate
import torch.nn as nn 
import torch
from torch_geometric.nn import  SAGEConv, HeteroConv, GATConv
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import *
import sys
import abc
import os
import tqdm
from torch_geometric.nn import MetaPath2Vec
from utils import *
import warnings
warnings.filterwarnings('ignore') 
    
class M2Vec(torch.nn.Module):
    
    def __init__(self, edge_index_dict, cfg):
        super().__init__()
        
        self.edge_index = edge_index_dict
        self.embeddgin_dimension = cfg.embedding_dim
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = MetaPath2Vec(edge_index_dict=edge_index_dict,**cfg).to(self._device)
        
        
    def _get_loader(self, batch_size:int=64, shuffle: bool=True):
        return self._model.loader(batch_size=batch_size, shuffle=shuffle)
    
    def fit(self,optimizer: Callable,batch_size:int, epochs:int, epsilon:float=1e-2):
        
        self._model.train()
        metapath_loader = self._get_loader(batch_size, shuffle=True)
        losses = []
        for epoch in tqdm.tqdm(range(epochs)):
            total_loss = 0
            for (pos_w, neg_w) in metapath_loader:
                optimizer.zero_grad()
                loss = self._model.loss(pos_w.to(self._device), neg_w.to(self._device))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            losses.append(total_loss)
            
            if epoch >= 10:
                if np.std(np.array(losses[-5:])) < epsilon:
                    print("Loss not converged")
                    return losses
        
        return losses
                
                
                
    def test(self, graph, batch_size:int= 32):
        self._model.eval()
        
        #Information to test
        nodes = (list(graph['attribute'].nodes.union(graph['entity'].nodes)))
        label_to_gt_at = {c:1 for c in (graph["attribute"].nodes)}
        label_to_gt_ent = {c:0 for c in (graph["entity"].nodes)}
        label_to_gt = {**label_to_gt_at, **label_to_gt_ent}
        gt = ([label_to_gt[i] for i in nodes])
        
        # make the shuffeling        
        nodes = torch.tensor(nodes).view(1,-1)
        gt = torch.tensor(gt).view(1,-1)
        
        to_test = torch.cat((nodes, gt), dim=0)
        perm = torch.randperm(to_test.shape[1])
        to_test = to_test[:, perm]
        ## fer la partició de manera adequada

        batch_nodes = np.array_split(to_test[0,:].numpy(), batch_size)
        gt_nodes = np.array_split(to_test[1,:].numpy(), batch_size)

        acc = []
        for batch_n, batch_gt in zip(batch_nodes, gt_nodes):
            # attributes
            att = batch_n[batch_gt == 1]
            z_att = self._model('attribute', batch=torch.tensor(att).to(self._device)).to(self._device)
            #entities
            ind = batch_n[batch_gt == 0]
            z_ind = self._model('entity', batch=torch.tensor(ind).to(self._device)).to(self._device)

            #temporal matrix to create new one 
            z = torch.zeros((batch_gt.shape[0], z_att.shape[1])).to(self._device)

            z[batch_gt == 1] = z_att
            z[batch_gt == 0] = z_ind

            perm = torch.randperm(z.shape[0])
            train_perm = perm[:int(z.shape[0] * 0.3)]
            test_perm = perm[int(z.shape[0] * 0.3):]

            batch_gt = torch.tensor(batch_gt).to(self._device)
            acc.append(self._model.test(z[train_perm], batch_gt[train_perm], z[test_perm],
                      batch_gt[test_perm], solver="newton-cg",max_iter=200))   
                
                
        return np.mean(np.array(acc))
    
    def transform(self, graph, kind:str="attribute",batch_size:int= 32):
        node_embeddings = graph.node_emb.to(self._device)

        indexes = np.array(list(graph[kind].nodes))
        np.random.shuffle(indexes)
        batches = np.array_split(indexes, batch_size)

        for batch in batches:
           batch = torch.tensor(batch).to(self._device)
           z = self._model(kind, batch=batch).to(self._device)
           node_embeddings[batch] = z

        return node_embeddings.cpu()

           
 
                
    
        
class Graph_Sage_GNN(torch.nn.Module):
    """GraphSAGE"""
    def __init__(self, cfg):
        super().__init__()

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        self.sage1 = SAGEConv((-1,-1), cfg.hidden_channels)
        self.sage2 = SAGEConv((-1,-1), cfg.out_channels)

        self._accuracy = lambda x,y: torch.sum(x==y)/x.shape[0]

        self._cosinus = torch.nn.CosineSimilarity(dim=1)

        self._layer = nn.Linear(in_features=cfg.out_channels*2, out_features=1)
        self._layer_norm = nn.LayerNorm(cfg.out_channels)
        self._sigmoide = nn.Sigmoid()


    def encode(self, x, edge_index):
        h = self.sage1(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5)
        h = self.sage2(h, edge_index)
        h= torch.relu(h)

        return self._layer_norm(h)#, self._sigmoide(self._cosinus(t1,t2))
    
    def decode(self, x, edge_index):
        
        h_final = torch.cat((torch.index_select(x, 0,edge_index[0]),torch.index_select(x, 0,edge_index[1])), dim=1)

        return self._sigmoide(self._layer(h_final))
       

    def forward(self, x, edge_index):
        h = self.encode(x=x, edge_index=edge_index)
        new_h = self.decode(x=h, edge_index=edge_index) 


    #h1 = torch.index_select()#h[edge_index[0]].detach()
    #h2 = #h[edge_index[1]].detach()

        return h, new_h#, self._sigmoide(self._cosinus(t1,t2))




    
