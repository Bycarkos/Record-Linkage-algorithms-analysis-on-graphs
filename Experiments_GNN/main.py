import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from logger.Metrics import *
from data.data_utils import batch_loader, define_embeddings
from data.ball_graph import ball_graph
from data.wordnet_graph import wordnet_graph
from utils import *
# in set up there is the ROOT DIR
from set_up import *

from torch_geometric.loader import NeighborLoader
from models.defined_models import *
from Experiments import Experiment_1, Experiment_2

import tqdm
import time
from termcolor import colored
import numpy as np
import os
import pickle


@hydra.main(config_path="./configs", config_name="train", version_base="1.1")
def main(cfg:DictConfig):

    print(cfg)
    # INIT TRACKER
    tracker = instantiate(cfg.logger.tracker)
    set_up_classic_track(create_file="run_log_files/"+cfg.logger.name_file)

    # init dataset
    name_graph = cfg.data.dataset["_target_"].split(".")[1]    
    if cfg.data.save == True:

        gr = instantiate(cfg.data.dataset)
        to_save = gr
        with open(os.path.join(ROOT_DIR, "Bashkar", name_graph), "wb") as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(os.path.join(ROOT_DIR, "Bashkar", name_graph), "rb") as handle:
            gr=pickle.load(handle)   
            
    #init params and dataset to setup model        
    cuda = cfg.setup.cuda
    epochs = cfg.setup.epochs
    embedding_size = cfg.setup.embedding_size
    batch = cfg.setup.batch
    edge_types = len(gr._r_to_idx)
    total_nodes = len(gr._nodes_to_idx)
    define_embeddings(graph=gr._graph, sample_size=len(gr._nodes_to_idx), feature_size=embedding_size)

    metapath = [tuple(['entity', 'kin', 'attribute']), tuple(['entity', 'job', 'attribute']), tuple(['entity', 'surname', 'attribute']), tuple(['entity', 'second_surname', 'attribute']) , tuple(['entity', 'name', 'attribute']), tuple(['entity', 'same_as', 'entity'])]
    cfg.models.m2vec["metapath"] = metapath
    cfg.models.m2vec["num_nodes_dict"] = {'entity': total_nodes , 'attribute': total_nodes}
    cfg.models.m2vec["embedding_dim"] = embedding_size

    cfg.models.graphSage["out_channels"] = embedding_size
    m2vec = M2Vec(edge_index_dict=gr.graph.edge_index_dict,cfg=cfg.models.m2vec) 

    # Metapath to vec space
    #with torch.autograd.profiler.profile(use_cuda=True,with_stack=True, profile_memory=True) as prof:

    if cfg.models.m2v.to_train != False:
        optimizer_m2v = instantiate(cfg.setup.optimizer, params=m2vec._model.parameters())
        losses_m2vec = m2vec.fit(optimizer=optimizer_m2v, batch_size=batch, epochs=epochs)
        save_model(m2vec._model.state_dict, os.path.join(ROOT_DIR, "model_save"),f"Model_{embedding_size}" )      

    elif cfg.models.m2v.trained == True:
        m2vec._model.load_state_dict(torch.load(os.path.join(ROOT_DIR, "model_save",f"Model_{embedding_size}")))
    
    # Testing between entity and attribute    
    acc_test_m2vec = m2vec.test(gr._graph, batch_size=batch)
    print(acc_test_m2vec)
    new_node_emb = m2vec.transform(gr._graph, batch_size=embedding_size)
    gr.graph.node_emb = new_node_emb

    ## Starting Graph Sage Block
    #loss
    criterion = torch.nn.BCELoss()

    #optimizer
    gnn = Graph_Sage_GNN(cfg.models.graphSage) 

    if cuda:gnn.to("cuda")

    graph_to_train = instantiate(cfg.models.graphSage.Experiment, data_object=gr).get_subgraph()
    graph_to_train.x = graph_to_train.node_emb

    print(graph_to_train)


if __name__ == "__main__":
    main()
