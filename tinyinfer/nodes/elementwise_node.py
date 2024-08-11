from ..params import *
import math
import torch
import numpy as np
import torch.nn.functional as F
from cuda import cudart
import kernels
from copy import deepcopy
from .base_node import Node


class ElementwiseNode(Node):
    def __init__(self):
        super().__init__()
        self.params = ElementwiseParams()
        self.type = "Add"
    
    def run(self, stream):
        in_edge0 = self.all_edges[self.input_names[0]]
        in_edge1 = self.all_edges[self.input_names[1]]
        out_edge = self.all_edges[self.output_names[0]]
        try:
            # print("****use cuda elementwise")
            kernels.elementwise(in_edge0.tensor.data_ptr(), in_edge1.tensor.data_ptr(), out_edge.tensor.data_ptr(),
                                in_edge0.shape, in_edge1.shape, out_edge.shape, self.network_precision,
                                "nchw", self.type, stream)
        except:
            # print("****use pytorch elementwise")
            if self.type == "Add":
                out_edge.tensor = in_edge0.tensor + in_edge1.tensor
            elif self.type == "Sub":
                out_edge.tensor = in_edge0.tensor - in_edge1.tensor
            elif self.type == "Mul":
                out_edge.tensor = in_edge0.tensor * in_edge1.tensor
            elif self.type == "Div":
                out_edge.tensor = in_edge0.tensor / in_edge1.tensor
            else :
                raise ModuleNotFoundError
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        n,c,h,w = in_edge.shape

        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = in_edge.shape
        if self.network_precision == "float32" :
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.network_precision == "float16" :
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] elementwise infer shape not support!!")
    
    def infer_layouts(self):
        pass