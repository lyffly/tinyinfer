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

class ReshapeNode(Node):
    def __init__(self):
        super().__init__()
        self.params = ReshapeParams()
        self.type = "Reshape"
        self.shape_to = []
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        # print("[Error] reshape run not impl")
        out_edge.tensor = in_edge.tensor.reshape(self.shape_to)
        
        
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        shape_edge = self.all_edges[self.input_names[1]]
        
        input_shape = list(in_edge.shape)
        input_data_length = 1
        for l in input_shape:
            input_data_length *= l
        shape_data = shape_edge.tensor.tolist()
        output_shape_tmp = deepcopy(shape_data)
        shape_data_length = 1
        index = -1
        index_1_count = 0
        for i,d in enumerate(shape_data):
            if d == -1:
                index = i
                index_1_count +=1
            else :
                shape_data_length *= d
        
        assert index_1_count == 1
        if index_1_count == 1 :
            output_shape_tmp[index] = input_data_length // shape_data_length
        
        self.shape_to = output_shape_tmp
        out_edge = self.all_edges[self.output_names[0]]
        if self.network_precision == "float32" :
            out_edge.shape = output_shape_tmp
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.network_precision == "float16" :
            out_edge.shape = output_shape_tmp
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] Reshape infer shape not support!!")
    
    def infer_layouts(self):
        pass