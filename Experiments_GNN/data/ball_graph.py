from data.data_utils import *
from data.Graph_Loader import *


from torch_geometric.typing import *
from typing import *
import os
from pathlib import Path
import torch



class ball_graph(Graph):

    def __init__(self, path: Path) -> None:
        super().__init__()

        # different graphs


        

        self._train_graph = self.read_data(path=path+"/train.tsv")
        self._test_graph = self.read_data(path=path+"/test.tsv")

        self._candidate_pairs = self.read_data(path=path+"/Candidate_Pairs_Test.tsv")

       ## Init all attributes to get the stats of the graph
        super().init_attributes()


        era = {"name", "second_surname", "surname", "kin", "job"}
        ere = (self._r_set - era)
        #ERAs
        self._era_graph = self._map_edges(partition="all", subgraph=era)  
        self._train_era_graph = self._map_edges(partition="train", subgraph=era)
        self._test_era_graph = self._map_edges(partition="test", subgraph=era)    

        #EREs
        self._ere_graph = self._map_edges(partition="all", subgraph=ere)  
        self._train_ere_graph = self._map_edges(partition="train", subgraph=ere)
        self._test_ere_graph = self._map_edges(partition="test", subgraph=ere) 


        self._init_ball()  





    def _init_ball(self):         

        # fill graph with the EA   
        self._fill_graph(edge_structure=self._era_graph, subgraph="era")
        # Fill graph with the EE
        self._fill_graph(edge_structure=self._ere_graph, subgraph="ere")       
        # Clean entities
        self._graph["entity"].nodes = (self._graph["entity"].nodes - self._graph["attribute"].nodes)
        self._graph["entity"].num_nodes = len(self._graph["entity"].nodes)        

        #Creating the masks fro the ERAs
        self._make_subgraph_partition(edge_structure=self._train_era_graph, subgraph="era", partition="train")
        self._make_subgraph_partition(edge_structure=self._test_era_graph, subgraph="era",partition="test")

        # Creating the masks for the EREs
        self._make_subgraph_partition(edge_structure=self._train_ere_graph, subgraph="ere" , partition="train")
        self._make_subgraph_partition(edge_structure=self._test_ere_graph, subgraph="era", partition="test")


    def _fill_graph(self,edge_structure: Dict[Tuple[str,str,str], List[int,]],subgraph:str) -> None:
        if subgraph == "era":
            src, trg = "entity", "attribute"

        elif subgraph == "ere":
           src, trg = "entity", "entity"
        else:
            raise AssertionError
        
        nd_type = trg
        attributes = set({})
        
        for e_idx, _ , trgts, edge_type,nodes in self._generate_adjency(edge_structure):
              
            self._graph[f"{src}", edge_type, f"{trg}"].edge_index = e_idx
            self._graph[f"{src}", edge_type, f"{trg}"].label = edge_type      
            self._graph[f"{src}", edge_type, f"{trg}"].type = self._r_to_idx[edge_type]
            attributes = attributes.union(nodes) if src == trg else attributes.union(set(trgts.view(-1).numpy()))

        self._graph[nd_type].nodes = attributes
        self._graph[nd_type].num_nodes = len(attributes)


    def _make_subgraph_partition(self, edge_structure: Dict[Tuple[str,str,str], List[int,]], subgraph:str ,partition="train"):
        
        if subgraph == "era":
            src, trg = "entity", "attribute"

        elif subgraph == "ere":
           src, trg = "entity", "entity"
        else:
            raise AssertionError

        for e_idx, _,_, edge_type, _ in self._generate_adjency(edge_structure):
    
            if partition == "train":
                self._graph[f"{src}", edge_type, f"{trg}"].train_edge_index = e_idx
            else:
                self._graph[f"{src}", edge_type, f"{trg}"].test_edge_index = e_idx

    def _generate_adjency(self, edge_structure: Dict[Tuple[str,str,str], List[int,]]):
        for edge_type, nodes in edge_structure.items():

            index_len = len(nodes)
            if index_len < 2:
                continue
            src_idx = torch.arange(0, index_len, step=2, dtype=torch.int32)
            trg_idx = torch.arange(1, index_len, step=2, dtype=torch.int32)
            
            tmp_tensor = torch.tensor(nodes).view(1,-1)
            srcs = torch.index_select(tmp_tensor, 1, src_idx)
            trgts = torch.index_select(tmp_tensor, 1, trg_idx)
            edge_idx = torch.cat((srcs, trgts), 0).unique(dim=1)         
        
            yield edge_idx, srcs, trgts, edge_type, nodes

    def _map_edges(self, partition:str="all", subgraph:set={}) -> dict:
        structure = self._map_classes_structure(subgraph)


        if partition == "all": edges = self._all_triples
        elif partition == "train": edges = self._train_graph
        else: edges = self._test_graph

        convert_s = self._nodes_to_idx
        convert_o = self._nodes_to_idx

        for s,r,o in edges:
            s_idx, r_idx, o_idx = convert_s[s], self._r_to_idx[r], convert_o[o]
            if not structure.get(r, None) is None:
                structure.get(r).extend([s_idx, o_idx])

        return structure


    def _map_classes_structure(self, Edge_types:Set[str]) -> dict:
        intersect = self._r_set & Edge_types
        return {r:[] for r in intersect}

    def _init_gt_by_task(self, task:str="prediction") -> None:

        if task.lower() == "prediction":
            s, o, gt = list(*zip(self._candidate_pairs))
            srcs = torch.tensor([self._nodes_to_idx[s_] for s_ in s])
            trg = torch.tensor([self._nodes_to_idx[o_] for o_ in o])
            gt = torch.tensor(gt)

            edge_index = torch.cat((srcs, trg), dim=0).unique(dim=1)

            self._graph["entity", "candidate_pair", "entity"].edge_index=edge_index
            self._graph["entity", "candidate_pair", "entity"].gt=gt
            
        
        elif task.lower() == "classification":
            for edge_type in self.graph.metadata()[1]:
                self._graph[edge_type].gt = self._graph[edge_type].type

        elif task.lower() == "mixed":
            for edge_type in self.graph.metadata()[1]:
                linked = 1 if edge_type[1] == "same_as" else 0
                self._graph[edge_type].gt = (self._graph[edge_type].type, linked)

        else:
            raise NotImplementedError

    def __generate_negsample(self,iterations:int):
        nodes = list(self._entity_set)

        for iter in range(iterations):
            src = np.random.choice(nodes)
            trg = np.random.choice(nodes)
            pair = [tuple([src, trg])]
            ppair = (set(pair) & self._total_edge_set) | ([tuple([trg,src])] & self._total_edge_set)
  
            if len(ppair) != 0:
                print(iter)
                while len(ppair) != 0:
                    trg = np.random.choice(nodes)
                    pair = [tuple([src, trg])]
                    ppair = (set(pair) & self._total_edge_set) | ([tuple([trg,src])] & self._total_edge_set)
                print("EXIT")

            pair_idx = [self._nodes_to_idx[src],self._nodes_to_idx[trg]]
            
            yield pair_idx
       
       

    def _add_negative_sample(self, negsamples:int=9):
        total_iterations = len(self._total_edge_set) * negsamples
        adj = []

        for neg_edge in self.__generate_negsample(total_iterations):
            adj.extend(neg_edge)



        index_len = len(adj)    
        src_idx = torch.arange(0, index_len, step=2, dtype=torch.int32)
        trg_idx = torch.arange(1, index_len, step=2, dtype=torch.int32)
        
        tmp_tensor = torch.tensor(adj).view(1,-1)
        srcs = torch.index_select(tmp_tensor, 1, src_idx)
        trgts = torch.index_select(tmp_tensor, 1, trg_idx)
        edge_idx = torch.cat((srcs, trgts), 0).unique(dim=1) 
            
        self._graph.neg_sample = edge_idx


    @staticmethod
    def read_data(path:Path) -> List[Tuple]:
        tsv_file = pd.read_csv(path, delimiter='\t', names=["src", "rel", "trg"])
        return [(str(s).strip(), '_'.join(x for x in str(r).strip().split("_") if not x.isnumeric() ).lower(), str(o).strip()) for idx, (s,r,o) in tsv_file.iterrows()]


    @property
    def graph(self):
        return self._graph

    def __str__(self):
        return "ball"

if __name__ == "__main__":
    
    graph = ball_graph(path=os.getcwd()+"/data/Bashkar")

    print(f"train distinct nodes = {len(graph._train_node_set)} ")
    print(f" val distinct nodes = {len(graph._val_node_set)} ")
    print(f" test distinct nodes =  {len(graph._test_node_set)}")
    
    
    print(f" The number of disjoined validation nodes = {len(graph._disjoined_node_val)}")
    print(f" The number of disjoined test nodes = {len(graph._disjoined_node_test)}")
    
    
    
    print(f"train distinct edges = {len(graph._train_edge_set)} ")
    print(f" val distinct edges = {len(graph._val_edge_set)} ")
    print(f" test distinct edges =  {len(graph._test_edge_set)}")
    
    
    print(f" The number of disjoined validation edges = {len(graph._disjoined_edge_val)}")
    print(f" The number of disjoined test edges = {len(graph._disjoined_edge_test)}")
