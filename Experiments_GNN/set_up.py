from logger.Metrics import AverageMetrics, Logger_Metrics, Sweep_Logger
from utils import create_folder_if_missing

from typing import Optional
import plotly.graph_objects as go
import torch
import logging
from colorama import Fore
import os




colors = {"DEBUG":Fore.BLUE, "INFO":Fore.CYAN,
          "WARNING":Fore.YELLOW, "ERROR":Fore.RED, "CRITICAL":Fore.MAGENTA}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.levelname in colors:
            msg = colors[record.levelname] + msg + Fore.RESET
        return msg


def loggingdecorator(name):
    logger = logging.getLogger(name)
    def _decor(fn):
        function_name = fn.__name__
        def _fn(*args, **kwargs):
            ret = fn(*args, **kwargs)
            argstr = [str(x) for x in args]
            argstr += [key+"="+str(val) for key,val in kwargs.items()]
            logger.debug("%s(%s) -> %s", function_name, ", ".join(argstr), ret)
            return ret
        return _fn
    return _decor



def set_up_classic_track(create_file:Optional[str]=None):
    logger = logging.getLogger("train")
    logger.setLevel(level=logging.DEBUG)

    test_logger = logging.getLogger("test")
    test_logger.setLevel(level=logging.DEBUG)

    if isinstance(create_file, str):
        create_folder_if_missing(os.path.join(os.getcwd(), create_file))
        
        fh = logging.FileHandler(os.path.join(os.getcwd(), create_file))
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(ColoredFormatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))

    handler  = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
    handler.setLevel(logging.DEBUG)
    
    if isinstance(create_file, str):
        logger.addHandler(fh)
        test_logger.addHandler(fh)

    #test
    test_logger.addHandler(handler)
    test_logger.setLevel(level=logging.DEBUG)
    
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger = logging.getLogger("train.iter")
    logger.setLevel(logging.DEBUG)










