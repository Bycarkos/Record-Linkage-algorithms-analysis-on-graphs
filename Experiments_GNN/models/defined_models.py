
import torch
import torch_geometric as tg
import torch_geometric.data as tgd
import torch_geometric.nn as tgf
import torch.nn as nn 
from torch_geometric.typing import Adj, OptTensor, Size
import numpy as np
import pandas as pd
from typing import *
import sys
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os 
from pathlib import Path
import abc

from torch_geometric.nn import MetaPath2Vec



class BaseModel(metaclass=abc.ABCMeta):
    def __init__(self, edge_index_dict, cfg):
        """
        Abstract class inherited by all models.

        :param edge_index: Adj.
        :param predicate_embeddings: (batch_size, predicate_embedding_size) Tensor.
        """
        self.edge_index = edge_index_dict
        self.embeddgin_dimension = cfg.embedding_dim
        self._model = None
    
    @abc.abstractmethod
    def _get_loader(self):
        raise NotImplementedError

    
class M2Vec(BaseModel):
    
    def __init__(self, edge_index_dict, cfg):
        super().__init__(edge_index_dict, cfg=cfg)
        
        
        self._model = MetaPath2Vec(edge_index_dict=edge_index_dict,**cfg)
        
        
    def _get_loader(self, batch_size:int=64, shuffle: bool=True):
        return self._model.loader(batch_size, shuffle)
    
        
class Graph_Sage_GNN(torch.nn.Module):
    
    def __init__(self, hidden_channels, out_channels, projection):
        super().__init__()
        self.conv1 = tgf.SAGEConv((-1, -1), hidden_channels, project=projection)
        self.conv2 = tgf.SAGEConv((-1, -1), out_channels, project=projection)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
