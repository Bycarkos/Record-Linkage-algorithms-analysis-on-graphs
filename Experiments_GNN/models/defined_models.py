from hydra.utils import instantiate
import torch
from torch_geometric.nn import  SAGEConv
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import *
import sys
import abc
import os
import tqdm
from torch_geometric.nn import MetaPath2Vec
from utils import save_model
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
    self.sage1 = SAGEConv((-1,-1), cfg.hidden_channels)
    self.sage2 = SAGEConv((-1,-1), cfg.out_channels)
    self.optimizer = instantiate(cfg.optimizer, params=self.parameters())
    
    self._accuracy = lambda x,y: torch.sum(x==y)/x.shape[0]
    
    
  def forward(self, x, edge_index):
    h = self.sage1(x, edge_index)
    h = torch.relu(h)
    h = F.dropout(h, p=0.5, training=self.training)
    h = self.sage2(h, edge_index)
    return h, F.log_softmax(h, dim=1)

  def fit(self, train_loader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = self.optimizer

    self.train()
    for epoch in range(epochs+1):
      acc = 0
      val_loss = 0
      val_acc = 0

      # Train on batches
      for batch in train_loader:
        optimizer.zero_grad()
        _, out = self(batch.x, batch.edge_index)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        acc += self._accuracy(out[batch.train_mask].argmax(dim=1), 
                        batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss += criterion(out[batch.val_mask], batch.y[batch.val_mask])
        val_acc += self._accuracy(out[batch.val_mask].argmax(dim=1), 
                            batch.y[batch.val_mask])

      # Print metrics every 10 epochs
      if(epoch % 10 == 0):
          print(f'Epoch {epoch:>3} | Train Loss: {loss/len(train_loader):.3f} '
                f'| Train Acc: {acc/len(train_loader)*100:>6.2f}% | Val Loss: '
                f'{val_loss/len(train_loader):.2f} | Val Acc: '
                f'{val_acc/len(train_loader)*100:.2f}%')
    
    
if __name__ == "__main__":
    m = Graph_Sage_GNN({"hidden_channels":120, "out_channels": 1, "projection":True})