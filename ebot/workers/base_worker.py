from abc import ABC, abstractmethod
from ebot.models import BaseModel


class BaseWorker(ABC):
    def __init__(self, model: BaseModel):
        self._model = model

    @property
    def model(self) -> BaseModel:
        return self._model

    @abstractmethod
    def aggregate(self, step: int):
        pass

