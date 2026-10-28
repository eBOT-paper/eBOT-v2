import time
import struct
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ctypes import cdll, cast, c_bool, c_int, c_void_p, c_char_p,\
    Structure, POINTER
from ebot.utils import evaluate, load_config, prepare_data
from ebot.workers import BaseWorker
from ebot.models import BaseModel

config = load_config("configs/local_config.json")

WORKER_NUM = config["worker_num"]
GRADIENT_SIZE = config["gradient_size"]
FRAGMENT_SIZE = config["fragment_size"]
SCALE_FACTOR = config["scale_factor"]
ifname = "ens3"


class AggregatorMap(Structure):
    _fields_ =  [
                    ("lock", c_bool), 
                    ("hcheck", c_bool * WORKER_NUM), 
                    ("lflag", c_bool), 
                    ("iter", c_int), 
                    ("childcnt", c_int), 
                    ("lgrads", c_int * GRADIENT_SIZE),
                    ("grads", c_int * GRADIENT_SIZE),
                ]


class EbpfWorker(BaseWorker):
    
    def __init__(self, model: BaseModel, clib_path: str = "./ebpf/agg_map_lib.so"):
        super().__init__(model)
        self._clib = cdll.LoadLibrary(clib_path)
        self._clib.get_aggmap.argtypes = [c_int, c_int]
        self._clib.get_aggmap.restype = AggregatorMap
        
        self._clib.busy_polling.argtypes = [c_int, c_int]
        self._clib.busy_polling.restype = POINTER(c_int) 

        self._clib.init_socket.argtypes = [c_char_p]
        self._clib.init_socket.restype = c_int
        
        self._clib.send_all_fragments.argtypes = [c_int, POINTER(c_int)]
        self._clib.send_all_fragments.restype = c_int

        # init
        self._clib.init_socket(ifname.encode("utf-8"))
        self._aggmap_fd = self._clib.get_aggmap_fd()

    def __del__(self):
        self._clib.close_socket()
    
    def setup(self, rank: int, world_size: int, master_ip: str, master_port: int) -> str:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_gradients(self, grads: torch.Tensor) -> torch.Tensor:
        grads *= SCALE_FACTOR
        grads = torch.clamp(grads, -2**31, 2**31 - 1)
        grads_int32 = grads.to(torch.int32)

        total_size = FRAGMENT_SIZE * GRADIENT_SIZE
        if grads_int32.numel() < total_size:
            pad_size = total_size - grads_int32.numel()
            grads_int32 = torch.nn.functional.pad(grads_int32, (0, pad_size))

        return grads_int32
        
    def pull(self, step: int) -> torch.Tensor:
        ptr = self._clib.busy_polling(self._aggmap_fd, step) # point to the same addr
        if not hasattr(self, "_grad_buf"):
            total_size = FRAGMENT_SIZE * GRADIENT_SIZE
            np_arr = np.ctypeslib.as_array(ptr, shape=(total_size,))
            np_arr = np_arr.view(np.int32)
            self._grad_buf = torch.from_numpy(np_arr)
        return self._grad_buf.to(torch.float32) / SCALE_FACTOR 
    
    def push(self, step: int, grads: torch.Tensor):
        ptr = grads.data_ptr()
        grads_c = cast(c_void_p(ptr), POINTER(c_int))
        self._clib.send_all_fragments(step, grads_c)

    def all_reduce(self, step: int, grads: torch.Tensor) -> torch.Tensor:
        grads_int32 = self.prepare_gradients(grads)
        self.push(step, grads_int32)
        return self.pull(step)

    def aggregate(self, step: int):
        grads_flat = self._model.get_gradients()
        agg_buf = self.all_reduce(step, grads_flat)
        agg_buf.div_(WORKER_NUM)
        self._model.set_gradients(agg_buf)


