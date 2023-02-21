import torch
import torch.nn as nn 
import numpy as np
import pandas as pd
from typing import *
from pathlib import Path




def read_triples(path:Path) -> List[Tuple]:
    tsv_file = pd.read_csv(path, delimiter='\t', names=["src", "rel", "trg"])
    return [(str(s).strip(), '_'.join(x for x in str(r).strip().split("_") if not x.isnumeric() ).lower(), str(o).strip()) for idx, (s,r,o) in tsv_file.iterrows()]


def read_raw(path:Path) -> List[Tuple]:
    triples = []
    with open(path, "r") as file:
        for line in file:
            s,r,o = line.strip().split()
            triples.append((str(s).strip(), '_'.join(x for x in str(r).strip().split("_") if not x.isnumeric() ).lower(), str(o).strip()))

    return triples
       
def make_batch_idx(size, batch_size):    
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res



def define_embeddings(graph, sample_size:int, feature_size:int=32) -> None:
    embeds = nn.Embedding(sample_size, feature_size)
    print(sample_size)
    graph.node_emb = torch.zeros(sample_size, feature_size)
    for nt in graph.node_types:
        look_up = torch.tensor([idx for idx in graph[nt].nodes], dtype=torch.long)
        x = embeds(look_up)
        graph[nt].x = x
        graph.node_emb[look_up] = x


def batch_loader(graph, batch_size_proportion:int=16) -> List[Tuple[int,int,int]]:

    train_mix_edges = []
    val_mix_edges = []
    test_mix_edges = []
    
    for edg in graph.metadata()[1]:
        edge = graph[edg]
                 

        #train_loader   
        adj_train = edge.get("train_edge_index", None)    
        gt_link_train = edge.get("train_link_gt", None)
        gt_class_train = edge.get("train_types", None)
        gate_train = torch.full_like(gt_class_train, fill_value=1)
        
        if adj_train is not None:
            train_mix_edges.extend(zip(adj_train[0].numpy(), adj_train[1].numpy(), gt_class_train.view(-1).numpy(), gt_link_train.view(-1).numpy(), gate_train.view(-1).numpy()))       
        
        #val_loader
        adj_val = edge.get("val_edge_index", None)               
        gt_link_val = edge.get("val_link_gt", None)
        gt_class_val = edge.get("val_types", None)
        gate_val = torch.full_like(gt_class_val, fill_value=-1)      
        
        if adj_val is not None:
            val_mix_edges.extend(zip(adj_val[0].numpy(), adj_val[1].numpy(), gt_class_val.view(-1).numpy(), gt_link_val.view(-1).numpy(), gate_val.view(-1).numpy()))       
            
        
        #test_loader
        adj_test = edge.get("test_edge_index", None)               
        gt_link_test = edge.get("test_link_gt", None)
        gt_class_test = edge.get("test_types", None)
        gate_test = torch.full_like(gt_class_test, fill_value=-1)
        
        if adj_test is not None:
            test_mix_edges.extend(zip(adj_test[0].numpy(), adj_test[1].numpy(), gt_class_test.view(-1).numpy(), gt_link_test.view(-1).numpy(), gate_test.view(-1).numpy()))    
                        
    test_mix_edges = test_mix_edges + train_mix_edges
    val_mix_edges = train_mix_edges + val_mix_edges 
    
    train_mix_edges = np.array(train_mix_edges)
    val_mix_edges = np.array(val_mix_edges)   
    test_mix_edges = np.array(test_mix_edges)   
       
    np.random.seed(3) 
    np.random.shuffle(train_mix_edges)   
    np.random.shuffle(val_mix_edges)   
    np.random.shuffle(test_mix_edges)   

    
    return np.array_split(train_mix_edges, batch_size_proportion),np.array_split(val_mix_edges, batch_size_proportion),np.array_split(test_mix_edges, batch_size_proportion)

 
    