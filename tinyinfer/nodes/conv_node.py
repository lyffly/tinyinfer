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


class ConvNode(Node):
    def __init__(self):
        super().__init__()
        self.params = ConvParams()
        self.type = "Conv"
        self.inc = 0
        self.outc = 0
        self.workspace_size = -1
        self.workspace_ptr = 0
        self.algo = -1
        self.support_layout = "nhwc"
        self.tmp_tensor_in = None
        self.tmp_tensor_out = None
        self.desc = None

    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        w_edge = self.all_edges[self.input_names[1]]
        b_edge = None
        if len(self.input_names) > 2:
            b_edge = self.all_edges[self.input_names[2]]
        out_edge = self.all_edges[self.output_names[0]]
        try:
            # print("\n****use cudnn conv")
            if self.algo < 0:
                # print("[Python] conv algo init : ", self.algo)
                self.algo = kernels.get_conv2d_algo(
                    self.params.kernel_shape,
                    self.params.pads,
                    self.params.strides,
                    self.params.dilations,
                    self.params.group,
                    in_edge.shape,
                    w_edge.shape,
                    b_edge.shape,
                    out_edge.shape,
                    self.op_precision,
                    self.support_layout,
                    stream,
                    self.desc,
                )
                assert self.algo >= 0
                # print("[Python] conv algo find : ", self.algo)
                self.workspace_size = kernels.get_conv2d_workspace_size(
                    self.params.kernel_shape,
                    self.params.pads,
                    self.params.strides,
                    self.params.dilations,
                    self.params.group,
                    in_edge.shape,
                    w_edge.shape,
                    b_edge.shape,
                    out_edge.shape,
                    self.op_precision,
                    self.support_layout,
                    self.algo,
                    stream,
                    self.desc,
                )
                assert self.workspace_size >= 0
                if self.workspace_size > 0:
                    _, self.workspace_ptr = cudart.cudaMalloc(self.workspace_size)
                # print("[Python] conv workspace size : ", self.workspace_size)

            if self.support_layout == "nhwc":
                kernels.layout_convert(
                    in_edge.tensor.data_ptr(),
                    self.tmp_tensor_in.data_ptr(),
                    in_edge.shape,
                    in_edge.shape,
                    self.op_precision,
                    "nchw",
                    "nhwc",
                    stream,
                )
                kernels.conv2d(
                    self.tmp_tensor_in.data_ptr(),
                    w_edge.tensor.data_ptr(),
                    b_edge.tensor.data_ptr(),
                    self.tmp_tensor_out.data_ptr(),
                    self.workspace_size,
                    self.workspace_ptr,
                    self.algo,
                    self.params.kernel_shape,
                    self.params.pads,
                    self.params.strides,
                    self.params.dilations,
                    self.params.group,
                    in_edge.shape,
                    w_edge.shape,
                    b_edge.shape,
                    out_edge.shape,
                    self.op_precision,
                    self.support_layout,
                    stream,
                    self.desc,
                )
                kernels.layout_convert(
                    self.tmp_tensor_out.data_ptr(),
                    out_edge.tensor.data_ptr(),
                    out_edge.shape,
                    out_edge.shape,
                    self.op_precision,
                    "nhwc",
                    "nchw",
                    stream,
                )
            elif self.support_layout == "nchw":
                kernels.conv2d(
                    in_edge.tensor.data_ptr(),
                    w_edge.tensor.data_ptr(),
                    b_edge.tensor.data_ptr(),
                    out_edge.tensor.data_ptr(),
                    self.workspace_size,
                    self.workspace_ptr,
                    self.algo,
                    self.params.kernel_shape,
                    self.params.pads,
                    self.params.strides,
                    self.params.dilations,
                    self.params.group,
                    in_edge.shape,
                    w_edge.shape,
                    b_edge.shape,
                    out_edge.shape,
                    self.op_precision,
                    "nchw",
                    stream,
                    self.desc,
                )

            # print("cudnn : ",out_edge.tensor[0][0][0][:10])
        except:
            raise IOError

    def infer_shapes(self):
        self.desc = kernels.create_conv2d_desc()
        in_edge = self.all_edges[self.input_names[0]]
        weights_edge = self.all_edges[self.input_names[1]]
        if len(self.input_names) > 2:
            bias_edge = self.all_edges[self.input_names[2]]
        n, c, h, w = in_edge.shape
        oc, c, kh, kw = weights_edge.shape
        self.inc = c
        self.outc = oc
        padh = (self.params.pads[0] + self.params.pads[2]) / 2
        padw = (self.params.pads[1] + self.params.pads[3]) / 2
        oh = math.floor(
            (h + 2 * padh - self.params.dilations[0] * (kh - 1) - 1)
            / self.params.strides[0]
            + 1
        )
        ow = math.floor(
            (w + 2 * padw - self.params.dilations[1] * (kw - 1) - 1)
            / self.params.strides[1]
            + 1
        )

        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = [n, oc, oh, ow]
        if self.op_precision == "float32":
            self.support_layout = "nchw"
        if self.op_precision == "float32":
            out_edge.dtype = "float32"
            ytensor = YTensor()
            ytensor.zeros(out_edge.shape, DataType.float32, DataLayout.nchw)
            ytensor.tensortype = TensorType.variable
            out_edge.tensor = ytensor
        elif self.op_precision == "float16":
            out_edge.dtype = "float16"
            weights_edge.tensor.half()
            if self.support_layout == "nhwc":
                weights_edge.tensor.convert_layout(DataLayout.nhwc)
            if bias_edge:
                bias_edge.tensor.half()
            ytensor = YTensor()
            ytensor.zeros(out_edge.shape, DataType.float16, DataLayout.nchw)
            ytensor.tensortype = TensorType.variable
            out_edge.tensor = ytensor
            # 临时用 额外的转换 fix later
            tmp_in_tensor = YTensor()
            tmp_in_tensor.zeros_like(in_edge.tensor)
            tmp_in_tensor.tensortype = TensorType.variable
            tmp_in_tensor.cuda()
            self.tmp_tensor_in = tmp_in_tensor
            tmp_out_tensor = YTensor()
            tmp_out_tensor.zeros_like(out_edge.tensor)
            tmp_out_tensor.tensortype = TensorType.variable
            tmp_out_tensor.cuda()
            self.tmp_tensor_out = tmp_out_tensor
        else:
            print("[Error] conv infer shape not support!!")

    def set_op_precision(self, dtype:str):
        self.op_precision = dtype
    
    def get_op_support_precision(self, precision):
        supported = ["float32", "float16"]
        if precision in supported:
            return True
        else:
            return False
