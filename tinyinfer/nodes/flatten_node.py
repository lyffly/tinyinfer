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
        # out_edge.tensor = in_edge.tensor.reshape(out_edge.shape)
        out_edge.tensor.set_data_ptr(in_edge.tensor.data_ptr(), True)

    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        n, c, h, w = in_edge.shape
        out_edge = self.all_edges[self.output_names[0]]
        if self.params.axis == 1 and self.network_precision == "float32":
            out_edge.shape = [n, c * h * w]
            out_edge.dtype = "float32"
            ytensor = YTensor()
            ytensor.zeros(out_edge.shape, DataType.float32, DataLayout.nchw)
            ytensor.tensortype = TensorType.variable
            out_edge.tensor = ytensor
        elif self.params.axis == 1 and self.network_precision == "float16":
            out_edge.shape = [n, c * h * w]
            out_edge.dtype = "float16"
            ytensor = YTensor()
            ytensor.zeros(out_edge.shape, DataType.float16, DataLayout.nchw)
            ytensor.tensortype = TensorType.variable
            out_edge.tensor = ytensor
        else:
            print("[Error] flatten infer shape not support!!")

    def infer_layouts(self):
        pass
