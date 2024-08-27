from ..params import *
import math
import torch
import numpy as np
import torch.nn.functional as F
from cuda import cudart
import kernels
from copy import deepcopy
from .base_node import Node
from kernels import YTensor, DataType, DataLayout, TensorType

class TransposeNode(Node):
    def __init__(self):
        super().__init__()
        self.params = TransposeParams()
        self.type = "Transpose"
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        #print("[Error] transpose run not impl")
        out_edge.tensor = torch.permute(in_edge.tensor, self.params.perm)
        
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        print(self.params.perm)
        
        in_edge_shape = in_edge.shape
        out_edge_shape_tmp = deepcopy(in_edge_shape)
        for i, shape in enumerate(out_edge_shape_tmp) :
            out_edge_shape_tmp[i] = in_edge_shape[self.params.perm[i]]
        
        out_edge = self.all_edges[self.output_names[0]]
        if self.network_precision == "float32" :
            out_edge.shape = out_edge_shape_tmp
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.network_precision == "float16" :
            out_edge.shape = out_edge_shape_tmp
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] Transpose infer shape not support!!")
    
    def infer_layouts(self):
        pass