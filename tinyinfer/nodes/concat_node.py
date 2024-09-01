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


class ConcatNode(Node):
    def __init__(self):
        super().__init__()
        self.params = ConcatParams()
        self.type = "Concat"

    def run(self, stream):
        out_edge = self.all_edges[self.output_names[0]]
        # print("[Error] concat run not impl")
        in_len = len(self.input_names)
        edges = []
        for i in range(in_len):
            edge_i = self.all_edges[self.input_names[i]]
            print(edge_i.tensor.shape)
            edges.append(edge_i.tensor)

        out_edge.tensor = torch.cat(edges, self.params.axis)

    def infer_shapes(self):
        in_len = len(self.input_names)
        axis = self.params.axis

        in_0_shape = list(self.all_edges[self.input_names[0]].shape)
        out_edge_shape_tmp = deepcopy(in_0_shape)
        out_edge_shape_tmp[axis] = 0

        for i in range(in_len):
            in_edge_shape = list(self.all_edges[self.input_names[i]].shape)
            out_edge_shape_tmp[axis] += in_edge_shape[axis]

        out_edge = self.all_edges[self.output_names[0]]
        if self.network_precision == "float32":
            out_edge.shape = out_edge_shape_tmp
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(
                out_edge.shape, dtype=torch.float32, requires_grad=False
            )
        elif self.network_precision == "float16":
            out_edge.shape = out_edge_shape_tmp
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(
                out_edge.shape, dtype=torch.float16, requires_grad=False
            )
        else:
            print("[Error] concat infer shape not support!!")

    def infer_layouts(self):
        pass
