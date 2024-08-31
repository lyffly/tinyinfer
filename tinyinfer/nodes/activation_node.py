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

class ActivationNode(Node):
    def __init__(self):
        super().__init__()
        self.params = ActivationParams()
        self.type = "Activation"
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        
        try:
            #print("****use cuda activation")
            kernels.activation(in_edge.tensor.data_ptr(), out_edge.tensor.data_ptr(), self.params.alpha,
                                self.params.beta, in_edge.shape, out_edge.shape, self.network_precision,
                                "nchw", self.type, stream)
        except:
            raise IOError
        
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        n,c,h,w = in_edge.shape

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
            print("[Error] activation infer shape not support!!")
    
    def infer_layouts(self):
        pass