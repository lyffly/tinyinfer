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
            # print("****use cuda activation")
            kernels.activation(
                in_edge.tensor.data_ptr(),
                out_edge.tensor.data_ptr(),
                self.params.alpha,
                self.params.beta,
                in_edge.shape,
                out_edge.shape,
                self.op_precision,
                "nchw",
                self.type,
                stream,
            )
        except:
            raise IOError

    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        n, c, h, w = in_edge.shape

        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = in_edge.shape
        if self.op_precision == "float32":
            out_edge.create(out_edge.shape, "float32")
        elif self.op_precision == "float16":
            out_edge.create(out_edge.shape, "float16")
        else:
            print("[Error] activation infer shape not support!!")

    def set_op_dtypes(self):
        pass
    
    def set_op_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = in_edge.shape
        pass
    
    def set_op_layout(self, layout):
        pass

    def set_op_precision(self, dtype:str):
        self.op_precision = dtype
    
    def get_op_support_precision(self, precision):
        supported = ["float32", "float16"]
        if precision in supported:
            return True
        else:
            return False
    
    def setup_op_tensors(self):
        pass
