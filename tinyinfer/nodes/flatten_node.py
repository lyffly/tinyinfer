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


class FlattenNode(Node):
    def __init__(self):
        super().__init__()
        self.params = FlattenParams()
        self.type = "Flatten"

    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.tensor.set_data_ptr(in_edge.tensor.data_ptr(), True, False)

    def setup_op_out_edges(self):
        out_edge = self.all_edges[self.output_names[0]]
        if self.params.axis == 1 and self.op_precision == "float32":
            out_edge.create(out_edge.shape, "float32")
        elif self.params.axis == 1 and self.op_precision == "float16":
            out_edge.create(out_edge.shape, "float16")
        else:
            print("[Error] flatten infer shape not support!!")

    def set_op_precision(self, dtype:str):
        supported = ["float32", "float16"]
        in_edge = self.all_edges[self.input_names[0]]
        if in_edge.dtype in supported :
            self.op_precision = dtype
        else :
            self.op_precision = in_edge.dtype
    
    
    def set_op_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        n, c, h, w = in_edge.shape
        if self.params.axis == 1:
            out_edge.set_shape([n, c * h * w])
        else:
            print("[Error] flatten infer shape not support!!")

    def set_op_max_shapes(self):
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.max_shape = out_edge.shape
