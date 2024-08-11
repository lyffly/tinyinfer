from ..params import *
import math
import torch
import numpy as np
import torch.nn.functional as F
from cuda import cudart
import kernels
from copy import deepcopy
from .base_node import Node

class SoftmaxNode(Node):
    def __init__(self):
        super().__init__()
        self.params = SoftmaxParams()
        self.type = "Softmax"
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        print("[Error] softmax run not impl")
        
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        n,c,h,w = in_edge.shape
        out_edge = self.all_edges[self.output_names[0]]
        if self.params.axis == 1 and self.network_precision == "float32" :
            out_edge.shape = []
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.params.axis == 1 and self.network_precision == "float16" :
            out_edge.shape = []
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] Softmax infer shape not support!!")
    
    def infer_layouts(self):
        pass