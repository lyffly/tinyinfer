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


class PoolNode(Node):
    def __init__(self):
        super().__init__()
        self.params = PoolParams()
        self.type = "pool"
        self.desc = None

    def run(self, stream):
        for name in self.input_names:
            edge = self.all_edges[name]
        for name in self.output_names:
            edge = self.all_edges[name]
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        if self.type == "GlobalAveragePool":
            n, c, h, w = in_edge.tensor.shape
            try:
                # print("****use cudnn GlobalAveragePool")
                kernels.pooling(
                    in_edge.tensor.data_ptr(),
                    out_edge.tensor.data_ptr(),
                    self.params.kernel_shape,
                    self.params.pads,
                    self.params.strides,
                    in_edge.shape,
                    out_edge.shape,
                    self.type,
                    self.op_precision,
                    "nchw",
                    stream,
                    self.desc,
                )
            except:
                raise IOError

        elif self.type == "MaxPool":
            try:
                kernels.pooling(
                    in_edge.tensor.data_ptr(),
                    out_edge.tensor.data_ptr(),
                    self.params.kernel_shape,
                    self.params.pads,
                    self.params.strides,
                    in_edge.shape,
                    out_edge.shape,
                    self.type,
                    self.op_precision,
                    "nchw",
                    stream,
                    self.desc,
                )
            except:
                raise IOError

    def setup_op_out_edges(self):
        self.desc = kernels.create_pooling_desc()
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        if self.type == "GlobalAveragePool":
            if self.op_precision == "float32":
                out_edge.create(out_edge.shape, "float32")
            elif self.op_precision == "float16":
                out_edge.create(out_edge.shape, "float16")
            else:
                print("[Error] avgpool infer shape not support!!")
        elif self.type == "MaxPool":
            if self.op_precision == "float32":
                out_edge.create(out_edge.shape, "float32")
            elif self.op_precision == "float16":
                out_edge.create(out_edge.shape, "float16")
            else:
                print("[Error] maxpool infer shape not support!!")
        else :
            print("[Error] pool type not support yet !!")
            raise IOError
        
        kernels.setup_pooling_descriptor(
            self.params.kernel_shape,
            self.params.pads,
            self.params.strides,
            in_edge.shape,
            out_edge.shape,
            self.type,
            self.op_precision,
            "nchw",
            self.desc,
        )

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
        n, c, h, w = in_edge.shape
        if self.type == "GlobalAveragePool":
            n, c, h, w = in_edge.tensor.shape
            out_edge.set_shape([n, c, 1, 1])
        elif self.type == "MaxPool":
            padh = self.params.pads[0]
            padw = self.params.pads[1]
            kh = self.params.kernel_shape[0]
            kw = self.params.kernel_shape[1]
            dilationh = 1
            dilationw = 1
            strideh = self.params.strides[0]
            stridew = self.params.strides[1]
            oh = math.floor((h + padh * 2 - ((kh - 1) * dilationh + 1)) / strideh + 1)
            ow = math.floor((w + padw * 2 - ((kw - 1) * dilationw + 1)) / stridew + 1)
            out_edge.set_shape([n, c, oh, ow])
        else :
            print("[Error] pool not support yet !!")
            raise IOError
