from ..params import *
import math
import torch
import numpy as np
import torch.nn.functional as F
from cuda import cudart
import kernels
from copy import deepcopy
from .base_node import Node

class GemmNode(Node):
    def __init__(self):
        super().__init__()
        self.params = GemmParams()
        self.type = "Gemm"
        self.workspace_size = 4096
        self.workspace_ptr = 0
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        w_edge = self.all_edges[self.input_names[1]]
        out_edge = self.all_edges[self.output_names[0]]
        bias_edge = None
        if len(self.input_names) > 2:
            bias_edge = self.all_edges[self.input_names[2]]
        
        try: # use cuda cublas
            if self.workspace_size : 
                _, self.workspace_ptr = cudart.cudaMalloc(self.workspace_size)
            if bias_edge:
                kernels.gemm(in_edge.tensor.data_ptr(), w_edge.tensor.data_ptr(), bias_edge.tensor.data_ptr(), out_edge.tensor.data_ptr(),
                        self.workspace_size, self.workspace_ptr, self.params.alpha, self.params.beta, 
                        self.params.transA, self.params.transB, in_edge.shape, w_edge.shape, bias_edge.shape,
                        out_edge.shape, self.network_precision, stream)
            else :
                kernels.gemm(in_edge.tensor.data_ptr(), w_edge.tensor.data_ptr(), 0, out_edge.tensor.data_ptr(),
                        self.workspace_size, self.workspace_ptr, self.params.alpha, self.params.beta, 
                        self.params.transA, self.params.transB, in_edge.shape, w_edge.shape, [], out_edge.shape, 
                        self.network_precision, stream)
            # print("****use cublas gemm\n")
        except: # use torch
            if self.params.transB:
                out_edge.tensor = torch.matmul(in_edge.tensor, w_edge.tensor.T)
            else:
                out_edge.tensor = torch.matmul(in_edge.tensor, w_edge)
            # print("****use pytorch gemm\n")
    
    def __del__(self):
        if self.workspace_ptr:
            try:
                cudart.cudaFree(self.workspace_ptr)
            except:
                pass
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        weights_edge = self.all_edges[self.input_names[1]]
        if len(self.input_names) > 2:
            bias_edge = self.all_edges[self.input_names[2]]
        m, k = in_edge.shape
        n = 0
        out_edge = self.all_edges[self.output_names[0]]
        if self.params.transB == 1: 
            n, _ = weights_edge.shape
        else :
            _, n = weights_edge.shape
        
        out_edge.shape = [m, n]
        if self.network_precision == "float32" :
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.network_precision == "float16" :
            out_edge.dtype = "float16"
            weights_edge.tensor = weights_edge.tensor.half()
            if bias_edge:
                bias_edge.tensor = bias_edge.tensor.half()
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] gemm infer shape not support!!")

    def infer_layouts(self):
        pass