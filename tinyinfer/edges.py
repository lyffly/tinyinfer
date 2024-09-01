import numpy as np
import torch


class Edge:
    def __init__(self):
        super().__init__()
        self.name = None
        self.is_constant = False
        self.shape = None  # [1,3,8,8]
        self.dtype = None  # float32 float16 int8 int32 bool
        self.data = None
        self.data_cpu = None
        self.tensor = None
        self.type = "constant"  # constant input output variable

    def prepare_data(self):
        pass
