import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ebot.workers import BaseWorker
from ebot.models import BaseModel


class TorchDDPWorker(BaseWorker):

    def __init__(self, model: BaseModel):
        super().__init__(model)

    def setup(self, rank: int, world_size: int, master_ip: str, master_port: int) -> str:
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method=f"tcp://{master_ip}:{master_port}",
            rank=rank,
            world_size=world_size
        )
        self._model = DistributedDataParallel(self._model)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def aggregate(self, step: int):
        pass
