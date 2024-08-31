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
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = in_edge.shape
        if self.network_precision == "float32" :
            out_edge.dtype = "float32"
            ytensor = YTensor()
            ytensor.zeros(out_edge.shape, DataType.float32, DataLayout.nchw)
            ytensor.tensortype = TensorType.variable
            out_edge.tensor = ytensor
        elif self.network_precision == "float16" :
            out_edge.dtype = "float16"
            ytensor = YTensor()
            ytensor.zeros(out_edge.shape, DataType.float16, DataLayout.nchw)
            ytensor.tensortype = TensorType.variable
            out_edge.tensor = ytensor
        else :
            print("[Error] Softmax infer shape not support!!")
    
    def infer_layouts(self):
        pass