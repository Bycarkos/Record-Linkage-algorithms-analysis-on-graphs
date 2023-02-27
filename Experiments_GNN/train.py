import torch.nn as nn 
import torch
from torch_geometric.nn import  SAGEConv, HeteroConv, GATConv
import torch.nn.functional as F
import numpy as np
from typing import *
import sys
import tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore') 


def fit(x, train_loader,net, cuda,optimizer, criterions):
    _cosinus = torch.nn.CosineSimilarity(dim=1)
    _accuracy = lambda x,y: torch.sum(x==y)/x.shape[0]
    torch.cuda.empty_cache()
    net.train()

    criterion_BCE = criterions[0]
    criterion_CE = criterions[1]

    acc = 0
    # Train on batches
    with torch.autograd.set_detect_anomaly(True):
        for batch in train_loader:
            
            
            src_index,trg_index,gt, gt_link = list(zip(*batch)) 


            gt = check_type(gt)
            gt = gt.type(torch.LongTensor)
            
            gt_link = check_type(gt_link)
            gt_link = gt_link.type(torch.float32)
            
            src_index = check_type(src_index)
            trg_index = check_type(trg_index)
            edge_index = torch.cat((src_index.view(1,-1), trg_index.view(1,-1)), dim=0)



            if cuda:
                if not check_isin_cuda(x): x = x.to("cuda")
                if not check_isin_cuda(edge_index): edge_index = edge_index.to("cuda")
                if not check_isin_cuda(gt): gt = gt.to("cuda")
                if not check_isin_cuda(gt_link): gt_link = gt_link.to("cuda")
                            
            node_embeddings, scores = net(x=x, edge_index=edge_index)#, scores = net(x=x, edge_index=edge_index)
            x = node_embeddings.detach()
            loss = criterion_BCE(scores, gt_link)
            predictions = torch.where(scores > 0.5, 1, 0)
            acc += _accuracy(predictions, gt_link)
            
            loss.backward()
            optimizer.step()
            
            optimizer.zero_grad()

    return x, acc, loss
    
    