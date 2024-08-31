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

class ResizeNode(Node):
    def __init__(self):
        super().__init__()
        self.params = ResizeParams()
        self.type = "Resize"
        self.size_to = []
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        scale_edge = self.all_edges[self.input_names[2]]
        out_edge = self.all_edges[self.output_names[0]]
        # print("[Error] resize run not impl")
        print(in_edge.shape)
        print(in_edge.tensor.shape)
        out_edge.tensor = F.interpolate(in_edge.tensor, size = self.size_to[2:])
        
    
    def infer_shapes(self):
        in_length = len(self.input_names)
        if in_length == 3:
            scale_edge = self.all_edges[self.input_names[2]]
        elif in_length == 4:
            scale_edge = self.all_edges[self.input_names[2]]
            sizes_edge = self.all_edges[self.input_names[3]]
        else:
            print("[Error] resize not support ")
        in_edge_shape = list(self.all_edges[self.input_names[0]].shape)
        out_edge_shape = deepcopy(in_edge_shape)
        scale_edge_shape = deepcopy(scale_edge.shape)
        for i, shape in enumerate(out_edge_shape):
            out_edge_shape[i] = int(shape * scale_edge.tensor[i])
        
        self.size_to = out_edge_shape
        out_edge = self.all_edges[self.output_names[0]]
        if self.network_precision == "float32" :
            out_edge.shape = out_edge_shape
            out_edge.dtype = "float32"
            ytensor = YTensor()
            ytensor.zeros(out_edge.shape, DataType.float32, DataLayout.nchw)
            ytensor.tensortype = TensorType.variable
            out_edge.tensor = ytensor
        elif self.network_precision == "float16" :
            out_edge.shape = out_edge_shape
            out_edge.dtype = "float16"
            ytensor = YTensor()
            ytensor.zeros(out_edge.shape, DataType.float16, DataLayout.nchw)
            ytensor.tensortype = TensorType.variable
            out_edge.tensor = ytensor
        else :
            print("[Error] resize infer shape not support!!")
    
    def infer_layouts(self):
        pass
