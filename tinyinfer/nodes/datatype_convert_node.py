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

class CastNode(Node):
    def __init__(self, in_dtype, out_dtype):
        super().__init__()
        self.params = CastParams()
        self.type = "Cast"
        self.in_dtype = in_dtype    #"float32"
        self.out_dtype = out_dtype  #"float16"
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        
        try: # use cuda cublas
            kernels.datatype_convert(in_edge.tensor.data_ptr(), out_edge.tensor.data_ptr(),
                        in_edge.shape, out_edge.shape, "nchw", self.in_dtype, self.out_dtype, stream)
            #print("****use cuda datatype_convert\n")
        except:
            if self.out_dtype == "float32":
                out_edge.tensor = in_edge.tensor.float()
            elif self.out_dtype == "float16":
                out_edge.tensor = in_edge.tensor.half()
            #print("****use pytorch datatype_convert\n")

    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        in_edge.dtype = self.in_dtype
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = in_edge.shape
        if self.out_dtype == "float32":
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.out_dtype == "float16":
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] Cast infer shape not support!!")

    def infer_layouts(self):
        pass