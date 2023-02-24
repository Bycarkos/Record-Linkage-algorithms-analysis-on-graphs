
import abc
from typing import *
from data.Graph_Loader import *


class Experiment(metaclass=abc.ABCMeta):
    
    def __init__(self, data_object:Type[Graph]):
        self._graph = data_object._graph
        self._nEdges = data_object._graph.edge_types
        self._era_edges_type = list(data_object._era_graph.keys())
        self._ere_edges_type = list(data_object._ere_graph.keys())
        
        print(self._era_edges_type)
        print(self._ere_edges_type)
        
    @abstractmethod
    def get_subgraph(self):
        raise NotImplementedError
    
    
class Experiment_1(Experiment):
    
    def __init__(self, data_object: Type[Graph]):
        super().__init__(data_object)
        self._discriminated_list = ["viu", "is_building"]
        
    def get_subgraph(self):

        subgraph_list = [edge for edge in self._nEdges if edge[1] not in self._discriminated_list]
        tmp = [edge for edge in self._nEdges if edge[1] in self._discriminated_list]
        return self._graph.edge_type_subgraph(subgraph_list)
    
    
    
class Experiment_2(Experiment):
    def __init__(self, data_object: Type[Graph]):
        super().__init__(data_object)
        
    def get_subgraph(self):
        return self._graph
        
        