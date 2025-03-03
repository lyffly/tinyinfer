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


class GemmNode(Node):
    def __init__(self):
        super().__init__()
        self.params = GemmParams()
        self.type = "Gemm"
        self.workspace_size = 4096
        self.workspace_ptr = 0

    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        w_edge = self.all_edges[self.input_names[1]]
        out_edge = self.all_edges[self.output_names[0]]
        bias_edge = None
        if len(self.input_names) > 2:
            bias_edge = self.all_edges[self.input_names[2]]

        try:  # use cuda cublas
            # if self.workspace_size :
            #     _, self.workspace_ptr = cudart.cudaMalloc(self.workspace_size)
            if bias_edge and in_edge.shape[0] == 1:
                kernels.gemv(
                    in_edge.tensor.data_ptr(),
                    w_edge.tensor.data_ptr(),
                    bias_edge.tensor.data_ptr(),
                    out_edge.tensor.data_ptr(),
                    self.workspace_size,
                    self.workspace_ptr,
                    self.params.alpha,
                    self.params.beta,
                    self.params.transA,
                    self.params.transB,
                    in_edge.shape,
                    w_edge.shape,
                    bias_edge.shape,
                    out_edge.shape,
                    self.op_precision,
                    stream,
                )

            elif bias_edge and in_edge.shape[0] > 1:
                kernels.gemm(
                    in_edge.tensor.data_ptr(),
                    w_edge.tensor.data_ptr(),
                    bias_edge.tensor.data_ptr(),
                    out_edge.tensor.data_ptr(),
                    self.workspace_size,
                    self.workspace_ptr,
                    self.params.alpha,
                    self.params.beta,
                    self.params.transA,
                    self.params.transB,
                    in_edge.shape,
                    w_edge.shape,
                    bias_edge.shape,
                    out_edge.shape,
                    self.op_precision,
                    stream,
                )
                # kernels.gemm_cutlass(in_edge.tensor.data_ptr(), w_edge.tensor.data_ptr(), bias_edge.tensor.data_ptr(), out_edge.tensor.data_ptr(),
                #         self.workspace_size, self.workspace_ptr, self.params.alpha, self.params.beta,
                #         self.params.transA, self.params.transB, in_edge.shape, w_edge.shape, bias_edge.shape,
                #         out_edge.shape, self.op_precision, stream)
            elif in_edge.shape[0] == 1:
                kernels.gemv(
                    in_edge.tensor.data_ptr(),
                    w_edge.tensor.data_ptr(),
                    0,
                    out_edge.tensor.data_ptr(),
                    self.workspace_size,
                    self.workspace_ptr,
                    self.params.alpha,
                    self.params.beta,
                    self.params.transA,
                    self.params.transB,
                    in_edge.shape,
                    w_edge.shape,
                    [],
                    out_edge.shape,
                    self.op_precision,
                    stream,
                )
            else:
                kernels.gemm(
                    in_edge.tensor.data_ptr(),
                    w_edge.tensor.data_ptr(),
                    0,
                    out_edge.tensor.data_ptr(),
                    self.workspace_size,
                    self.workspace_ptr,
                    self.params.alpha,
                    self.params.beta,
                    self.params.transA,
                    self.params.transB,
                    in_edge.shape,
                    w_edge.shape,
                    [],
                    out_edge.shape,
                    self.op_precision,
                    stream,
                )
        except:
            raise IOError

    def __del__(self):
        if self.workspace_ptr:
            try:
                cudart.cudaFree(self.workspace_ptr)
            except:
                pass

    def setup_op_out_edges(self):
        weights_edge = self.all_edges[self.input_names[1]]
        if len(self.input_names) > 2:
            bias_edge = self.all_edges[self.input_names[2]]
        out_edge = self.all_edges[self.output_names[0]]

        if self.op_precision == "float32":
            out_edge.create(out_edge.shape, "float32")
        elif self.op_precision == "float16":
            out_edge.create(out_edge.shape, "float16")
            weights_edge.tensor.half()
            if bias_edge:
                bias_edge.tensor.half()
        else:
            print("[Error] gemm infer shape not support!!")

    def set_op_precision(self, dtype: str):
        supported = ["float32", "float16"]
        in_edge = self.all_edges[self.input_names[0]]
        if in_edge.dtype in supported:
            self.op_precision = dtype
        else:
            self.op_precision = in_edge.dtype

    def get_op_support_precision(self, precision):
        supported = ["float32", "float16"]
        if precision in supported:
            return True
        else:
            return False

    def set_op_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        weights_edge = self.all_edges[self.input_names[1]]
        m, k = in_edge.shape
        n = 0
        if self.params.transB == 1:
            n, _ = weights_edge.shape
        else:
            _, n = weights_edge.shape
        out_edge.set_shape([m, n])

    def set_op_max_shapes(self):
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.max_shape = out_edge.shape
