import torch
import wandb
from typing import *
from torch_geometric.typing import Adj, OptTensor, Size

class AverageMetrics(object):


    def __init__(self):

        self._reset_params()

    def _reset_params(self):

        self._accuracy_events = []

        self._precision_events = []

        self._recall_events = []

        self.loss = 0
        self._loss_events = []
        

    def _update_recall(self, recall):
        self._recall_events.append(recall)

    def _update_precision(self, precision):
        self._precision_events.append(precision)

    def _update_accuracy(self, accuracy):
        self._accuracy_events.append(accuracy)

    def _compute_f_score(self):
        assert len(self._recall_events) == len(self._precision_events), f"{len(self._recall_events)}, {len(self._precision_events)}"

        numerator = 2*torch.mul(torch.tensor(self._recall_events), torch.tensor(self._precision_events))
        denominator = torch.add(torch.tensor(self._recall_events), torch.tensor(self._precision_events))

        return torch.div(numerator,denominator)    
    
    def _update_loss(self, loss):
        self.loss += loss
        self._loss_events.append(loss)


    def __name__():
        return "Average"

class Logger_Metrics(object):

    def __init__(self, cfg):
            wandb.Api(timeout=100)
            wandb.login()

            self._wdb = wandb.init(**cfg)
            self._wdb.define_metric("train/step")
            self._wdb.define_metric("val/step")
            self._wdb.define_metric("test/step")
            
            self._wdb.define_metric("train/*", step_metric="train/step")
            self._wdb.define_metric("val/*", step_metric="val/step")
            self._wdb.define_metric("test/*", step_metric="test/step")

            assert isinstance(self._wdb, type(wandb.run)), "Something went wrong with wandb connection"


    # Per veure la capacitat inductiva del codi
    def log_prediction_table(self, clas: List[str], predicted:List[str], is_in_train:List[Tuple[bool,bool]], probs:torch.FloatTensor, num_classes:int = 58, table_partition:str="train") -> None:
            # 🐝 Create a wandb Table to log images, labels and predictions to
        table = wandb.Table(columns=["original", "pred", "src_trained", "trg_trained"]+[f"score_{i}" for i in range(num_classes)])
            
        for clas, pred, trained, prob in zip(clas, predicted, is_in_train, probs):
            table.add_data(clas, pred, trained[0], trained[1], *prob)

        self._wdb.log({table_partition:table}, commit=False)

    def log_metrics(self, metrics:dict):
        self._wdb.log(metrics)

    def log_params(self, params:dict):
        table = wandb.Table(columns=list(params.keys()))   
         
        par = list(params.values())

        table.add_data(*par)
        self._wdb.log({"params":table}, commit=False)
        
        
    def save_final_comparations(self):
        pass

    def watch_model(self, model):
        self._wdb.watch(model)

    def define_alert(self, title:str, e: str):
        self._wdb.alert(title=title, text=e)
        

    def _close(self):
        self._wdb.finish()


    def __name__():
        return "W&b"


class Sweep_Logger(Logger_Metrics):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._reset_config()

    def _reset_config(self):
        self._sweep_config = {}

    # It can be maximize if the goal is (accuracy, recall, fscore etc etc)
    def define_goal(self, goal: str = "minimize", metric:str = "loss"):
        self._sweep_config["metric"] = {"name": metric, "goal": goal}

    # Choice among [grid, random, bayesian]
    def definem_method(self, method: str = "bayesian"):
        """
        grid Search:  Iterate over every combination of hyperparameter values. Very effective, but can be computationally costly.
*       
        random Search: Select each new combination at random according to provided `distribution`s. Surprisingly effective!
        
        bayesian Search: Create a probabilistic model of metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric. 
                        Works well for small numbers of continuous parameters but scales poorly.
        """

        self._sweep_config["method"] = method

    def define_sweep_config(self, parameters_config:dict):

        """
            example: parameters_dict = {
                    'optimizer': {
                        'values': ['adam', 'sgd']
                        },
                    'fc_layer_size': {
                        'values': [128, 256, 512]
                        },
                    'dropout': {
                        'values': [0.3, 0.4, 0.5]
                        },
                    'epochs': {'value': 1},
                        'learning_rate': {
                        # a flat distribution between 0 and 0.1
                        'distribution': 'uniform',
                        'min': 0,
                        'max': 0.1
                    },
                    'batch_size': {
                        # integers between 32 and 256
                        # with evenly-distributed logarithms 
                        'distribution': 'q_log_uniform_values',
                        'q': 8,
                        'min': 32,
                        'max': 256,
                    }
                    }
        """
        self._sweep_config["parameters"] = parameters_config
        
    def _init_train_sweep(self, train: Type[Callable], epochs:int=50):
        sweep_id = self._wdb.sweep(self._sweep_config, project=self._project)

        self._wdb.agent(sweep_id, train, count=epochs)


    def _close(self):
        super().__close()


    def __name__(self):
        return "Sweep"



