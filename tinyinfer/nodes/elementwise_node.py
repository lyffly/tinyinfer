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

    def set_op_precision(self, dtype: str):
        supported = ["float32", "float16"]
        in_edge = self.all_edges[self.input_names[0]]
        if in_edge.dtype in supported:
            self.op_precision = dtype
        else:
            self.op_precision = in_edge.dtype

    def set_op_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.set_shape(in_edge.shape)

    def set_op_max_shapes(self):
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.max_shape = out_edge.shape
