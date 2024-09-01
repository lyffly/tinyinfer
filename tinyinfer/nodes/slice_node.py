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


class SliceNode(Node):
    def __init__(self):
        super().__init__()
        self.params = SliceParams()
        self.type = "Slice"
        self.starts = []
        self.axes = []
        self.steps = []
        self.ends = []

    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        print("[Error] slice run not impl")

    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        starts_edges = self.all_edges[self.input_names[1]]
        ends_edges = self.all_edges[self.input_names[2]]
        axes_edges = self.all_edges[self.input_names[3]]
        steps_edges = self.all_edges[self.input_names[4]]
        out_tmp_shapes = deepcopy(list(in_edge.shape))
        starts = starts_edges.tensor
        ends = ends_edges.tensor
        axes = axes_edges.tensor
        steps = steps_edges.tensor
        for i, axis in enumerate(axes):
            start = starts[i]
            end = (
                int(ends[i])
                if int(ends[i]) < out_tmp_shapes[axis]
                else out_tmp_shapes[axis]
            )
            step = steps[i]
            out_tmp_shapes[axis] = int((end - start + 1) // step)
            self.starts.append(start)
            self.axes.append(axis)
            self.steps.append(step)
            self.ends.append(end)
        # print(out_tmp_shapes)

        out_edge = self.all_edges[self.output_names[0]]
        if self.network_precision == "float32":
            out_edge.shape = out_tmp_shapes
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(
                out_edge.shape, dtype=torch.float32, requires_grad=False
            )
        elif self.network_precision == "float16":
            out_edge.shape = out_tmp_shapes
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(
                out_edge.shape, dtype=torch.float16, requires_grad=False
            )
        else:
            print("[Error] Slice infer shape not support!!")

    def infer_layouts(self):
        pass
