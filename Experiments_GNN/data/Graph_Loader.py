from data.data_utils import *


from typing import *
from pathlib import Path
import torch_geometric.data as tgd
from abc import ABC, abstractmethod, abstractproperty
import os


class Graph(ABC):

    def __init__(self) -> None:
        super().__init__()
        self._graph = tgd.HeteroData()
        self._train_graph = None
        self._val_graph  = None
        self._test_graph = None        
        


    def init_attributes(self):

        self._all_triples = self._train_graph+self._test_graph
        self._all_link_gt = self._train_link_gt + self._test_link_gt + self._candidate_link_gt
        
        ### partition set nodes
        self._train_node_set = {s for (s, p, o) in self._train_graph} | {o for (s, p, o) in self._train_graph}
        self._test_node_set = {s for (s, p, o) in self._test_graph} | {o for (s, p, o) in self._test_graph}
    
        self._disjoined_node_test = self._test_node_set - self._train_node_set

        ### partition set edges
        self._train_edge_set = {(s,o) for (s, p, o) in self._train_graph} 
        self._test_edge_set = {(s,o) for (s, p, o) in self._test_graph}

        self._disjoined_edge_test = self._test_edge_set - self._train_edge_set       

        ### Splited sets
        self._s_set = {s for (s,p,o) in self._all_triples}
        self._r_set = {p for (s,p,o) in self._all_triples}
        self._o_set = {o for (s,p,o) in self._all_triples}
        
        ### joined sets
        self._entity_set = {s for (s, p, o) in self._all_triples} | {o for (s, p, o) in self._all_triples}
        self._total_edge_set = self._train_edge_set  | self._test_edge_set


        ### splitted Ordinal encoding
        self._s_to_idx = {s:idx for idx,s in enumerate(sorted(self._s_set))}
        self._idx_to_s = {value: key for key, value in self._s_to_idx.items()}

        self._r_to_idx = {p:idx for idx,p in enumerate(sorted(self._r_set))}
        self._idx_to_r = { value: key for key, value in self._r_to_idx.items()}

        self._o_to_idx = {o:idx for idx,o in enumerate(sorted(self._o_set))}
        self._idx_to_o = { value: key for key, value in self._o_to_idx.items()}       
        
        ### Joined Ordinal Encoding
        self._nodes_to_idx = {entity: idx for idx, entity in enumerate(sorted(self._entity_set))}
        self._idx_to_nodes = { value: key for key, value in self._nodes_to_idx.items()} 


    def save_graph(self):
        pass

    def load_graph(self):
        pass


    @abstractmethod
    def _map_edges(self):
        raise NotImplementedError


    @abstractmethod
    def _make_subgraph_partition(self):
        raise NotImplementedError

    @abstractmethod
    def read_data(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def graph(self):
        return self._graph


    @abstractmethod
    def _fill_graph(self):
        raise NotImplementedError


    @abstractmethod
    def _add_negative_sample(self):
        raise NotImplementedError


    