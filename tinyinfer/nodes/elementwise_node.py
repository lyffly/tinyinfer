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
            kernels.elementwise(
                in_edge0.tensor.data_ptr(),
                in_edge1.tensor.data_ptr(),
                out_edge.tensor.data_ptr(),
                in_edge0.shape,
                in_edge1.shape,
                out_edge.shape,
                self.op_precision,
                "nchw",
                self.type,
                stream,
            )
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
            else:
                raise ModuleNotFoundError

    def setup_op_out_edges(self):
        out_edge = self.all_edges[self.output_names[0]]
        if self.op_precision == "float32":
            out_edge.create(out_edge.shape, "float32")
        elif self.op_precision == "float16":
            out_edge.create(out_edge.shape, "float16")
        else:
            print("[Error] elementwise infer shape not support!!")

    def set_op_precision(self, dtype:str):
        self.op_precision = dtype
    
    def get_op_support_precision(self, precision):
        supported = ["float32", "float16"]
        if precision in supported:
            return True
        else:
            return False
    
    def set_op_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.set_shape(in_edge.shape)
