import torch.nn as nn
from abc import abstractmethod, ABC

class BaseLinearModule(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def from_pseudo(self):
        raise NotImplementedError
