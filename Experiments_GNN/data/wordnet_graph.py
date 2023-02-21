from data.data_utils import *
from data.Graph_Loader import *


from typing import *
import os
from pathlib import Path
import torch



class wordnet_graph(Graph):

    def __init__(self, path: Path) -> None:
        super().__init__()

        self._train_graph, self._train_link_gt = self.read_data(path=os.path.join(path, "train", "train-labeled.txt"))
        self._val_graph, self._val_link_gt = self.read_data(path=os.path.join(path, "train", "valid-labeled.txt"))

        tests = os.listdir(path=os.path.join(path, "test", "test-random-sample"))
        self._test_graph, self._test_link_gt = [],[]

        for tt in tests:
            mt, gt = self.read_data(path=os.path.join(path, "test", "test-random-sample", tt))
            self._test_graph.extend(mt)
            self._test_link_gt.extend(gt) 

        ## Init all attributes to get the stats of the graph
        super().init_attributes()

        self._init_wn()


    def _init_wn(self):
            
            ## Creating the structures of the adkj matrix
            self._graph_structure = self._map_edges(partition="all")  
            self._train_graph_structure = self._map_edges(partition="train")
            self._val_graph_structure = self._map_edges(partition="val")
            self._test_graph_structure = self._map_edges(partition="test")       
            
            # making the graph
            self._fill_graph(edge_structure=self._graph_structure)
            self._graph["entity"].nodes = (self._graph["entity"].nodes - self._graph["attribute"].nodes)
            self._graph["entity"].num_nodes = len(self._graph["entity"].nodes)


            #Creating the masks
            self._make_subgraph_partition(edge_structure=self._train_graph_structure, partition="train")
            self._make_subgraph_partition(edge_structure=self._val_graph_structure, partition="val")
            self._make_subgraph_partition(edge_structure=self._test_graph_structure, partition="test")

    def _fill_graph(self, edge_structure: Dict[Tuple[str,str,str], List[int,]]):
        

        src, trg = ("entity", "attribute")
        attributes = set({})
        entities = set({})

        for e_idx, srcs,trgts, edge_type, gts in self._generate_adjency(edge_structure):
            types = torch.full_like(gts,1) & gts
            self._graph[f"{src}", edge_type, f"{trg}"].edge_index = e_idx
            self._graph[f"{src}", edge_type, f"{trg}"].label = edge_type      
            self._graph[f"{src}", edge_type, f"{trg}"].type = torch.where(types == 1, self._r_to_idx[edge_type], len(self._r_set))
            self._graph[f"{src}", edge_type, f"{trg}"].gt = gts


            attributes =  attributes.union(set(trgts.view(-1).numpy()))
            entities = entities.union(set(srcs.view(-1).numpy()))

            self._graph[trg].nodes = attributes
            self._graph[trg].num_nodes = len(attributes)
            self._graph[src].nodes = entities


    def _make_subgraph_partition(self, edge_structure: Dict[Tuple[str,str,str], List[int,]], partition="train"):
        
        src, trg = ("entity", "attribute")

        for e_idx, _,_, edge_type,gts in self._generate_adjency(edge_structure):
            types = torch.full_like(gts,1) & gts

            if partition == "train":
                self._graph[f"{src}", edge_type, f"{trg}"].train_edge_index = e_idx
                self._graph[f"{src}", edge_type, f"{trg}"].train_link_gt = gts
                self._graph[f"{src}", edge_type, f"{trg}"].train_types = torch.where(types == 1, self._r_to_idx[edge_type], len(self._r_set))

            elif partition == "val":
                self._graph[f"{src}", edge_type, f"{trg}"].val_edge_index = e_idx
                self._graph[f"{src}", edge_type, f"{trg}"].val_link_gt = gts
                self._graph[f"{src}", edge_type, f"{trg}"].val_types = torch.where(types == 1, self._r_to_idx[edge_type], len(self._r_set))

            else:
                self._graph[f"{src}", edge_type, f"{trg}"].test_edge_index = e_idx
                self._graph[f"{src}", edge_type, f"{trg}"].test_link_gt = gts
                self._graph[f"{src}", edge_type, f"{trg}"].test_types = torch.where(types == 1, self._r_to_idx[edge_type], len(self._r_set))


    def _generate_adjency(self, edge_structure: Dict[Tuple[str,str,str], List[int,]]):
        for edge_type, nodes in edge_structure.items():

            index_len = len(nodes)
            src_idx = torch.arange(0, index_len, step=3, dtype=torch.int32)
            trg_idx = torch.arange(1, index_len, step=3, dtype=torch.int32)
            gt_idx = torch.arange(2, index_len, step=3, dtype=torch.int32)
            
            tmp_tensor = torch.tensor(nodes).view(1,-1)
            srcs = torch.index_select(tmp_tensor, 1, src_idx)
            trgts = torch.index_select(tmp_tensor, 1, trg_idx)
            gts = torch.index_select(tmp_tensor, 1, gt_idx)
            edge_idx = torch.cat((srcs, trgts), 0)        
        
            yield edge_idx, srcs, trgts, edge_type, gts


    def _map_edges(self, partition:str="all") -> dict:
        structure = {}

        if partition == "all": 
            edges = self._all_triples
            gt = self._all_link_gt
        elif partition == "train":
            edges = self._train_graph
            gt = self._train_link_gt
        elif partition  == "val": 
            edges = self._val_graph
            gt = self._val_link_gt

        else: 
            edges = self._test_graph
            gt = self._test_link_gt

        convert_s = self._nodes_to_idx
        convert_o = self._nodes_to_idx
        for (s,r,o), g in zip(edges, gt):
            s_idx, r_idx, o_idx = convert_s[s], self._r_to_idx[r], convert_o[o]
            if structure.get(r, None) is None:
                structure[r] = [s_idx, o_idx,g]
            else:
                structure.get(r).extend([s_idx, o_idx,g])


        return structure
        
    def __generate_negsample(self,iterations:int, nodes:list, edges:set):

        for iter in range(iterations):
            src = np.random.choice(nodes)
            trg = np.random.choice(nodes)
            pair = [tuple([src, trg])]
            ppair = (set(pair) & edges) | (set([tuple([trg,src])]) & edges)
  
            if len(ppair) != 0:
                print(iter)
                while len(ppair) != 0:
                    trg = np.random.choice(nodes)
                    pair = [tuple([src, trg])]
                    ppair = (set(pair) & edges) | (set([tuple([trg,src])]) & edges)
                print("EXIT")

            pair_idx = [self._nodes_to_idx[src],self._nodes_to_idx[trg]]
            
            yield pair_idx
       

    def _add_negative_sample(self, negsamples:int=9, partition:str="train"):
        adj = []
        if partition == "train":
            nodes = list(self._train_node_set)
            edges = self._train_edge_set
            total_iterations = len(edges) * negsamples

        elif partition == "val":
            nodes = list(self._val_node_set)
            edges = self._val_edge_set
            total_iterations = len(edges) * negsamples
            
        else:
            nodes = list(self._test_node_set)
            edges = self._test_edge_set
            total_iterations = len(edges) * negsamples
            
        for neg_edge in self.__generate_negsample(total_iterations, nodes, edges):
            adj.extend(neg_edge)

        index_len = len(adj)    
        src_idx = torch.arange(0, index_len, step=2, dtype=torch.int32)
        trg_idx = torch.arange(1, index_len, step=2, dtype=torch.int32)
        
        tmp_tensor = torch.tensor(adj).view(1,-1)
        srcs = torch.index_select(tmp_tensor, 1, src_idx)
        trgts = torch.index_select(tmp_tensor, 1, trg_idx)
        edge_idx = torch.cat((srcs, trgts), 0).unique(dim=1) 


        if partition == "train":
            self._graph["entity", "neg", "attribute"].train_edge_index = edge_idx
        elif partition == "val":
            self._graph["entity", "neg", "attribute"].val_edge_index = edge_idx
        else:
            self._graph["entity", "neg", "attribute"].test_edge_index = edge_idx

            
        
                       

    @staticmethod
    def read_data(path:Path) -> List[Tuple]:
        triples = []
        link_pred = []
        with open(path, "r") as file:
            for line in file:
                val = line.strip().split()
                s,r,o = val[0], val[1], val[2:]
                if len(o)>1:
                    o,g = o[0], o[1]
                    link_pred.append(int(g))
                
                triples.append((str(s).strip(), '_'.join(x for x in str(r).strip().split("_") if not x.isnumeric() ).lower(), str(o).strip()))

        return triples, link_pred

    @property
    def graph(self):
        return self._graph

    def __str__(self):
        return "wn18"
        
if __name__ == "__main__":
    
    ## TODO MIRAR AIXÔ MILLOR 
    graph = wordnet_graph(path=os.getcwd()+"/data/GraIL-BM_WN18RR_v4")

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


