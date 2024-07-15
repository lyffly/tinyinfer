import numpy as np
import torch

class Edge:
    def __init__(self):
        super().__init__()
        self.name = None
        self.is_constant = False
        self.shape = None
        self.dtype = None
        self.data = None
        self.data_cpu = None
        self.tensor = None
        self.type = "constant" # constant input output variable
    
    def prepare_data(self):
        
        pass


