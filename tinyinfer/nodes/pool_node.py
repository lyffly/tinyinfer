from ..params import *
import math
import torch
import numpy as np
import torch.nn.functional as F
from cuda import cudart
import kernels
from copy import deepcopy
from .base_node import Node

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
            n,c,h,w = in_edge.tensor.shape
            # (int64_t in_ptr, int64_t out_ptr, std::vector<int> kernels,
            #          std::vector<int> paddings, std::vector<int> strides, std::vector<int> in_shape,
            #          std::vector<int> out_shape, std::string optype, std::string dtype,
            #          std::string layout, int64_t pstream, void* desc)
            try:
                #print("****use cudnn GlobalAveragePool")
                kernels.pooling(in_edge.tensor.data_ptr(),  out_edge.tensor.data_ptr(), self.params.kernel_shape,
                                self.params.pads, self.params.strides, in_edge.shape, out_edge.shape, self.type,
                                self.network_precision, "nchw", stream, self.desc)
            except:
                out_edge.tensor = F.avg_pool2d(in_edge.tensor,(h,w))
            
        elif self.type == "MaxPool":
            if True:
                print("****use cudnn MaxPool")
                # print(in_edge.tensor)
                print(in_edge.tensor.shape)
                print(out_edge.tensor.shape)
                print(self.type)
                kernels.pooling(in_edge.tensor.data_ptr(),  out_edge.tensor.data_ptr(), self.params.kernel_shape,
                                self.params.pads, self.params.strides, in_edge.shape, out_edge.shape, self.type,
                                self.network_precision, "nchw", stream, self.desc)
            # except:
            #     out_edge.tensor = F.max_pool2d(in_edge.tensor, self.params.kernel_shape,
            #                             stride=self.params.strides,padding=self.params.pads[:2])


    def infer_shapes(self):
        self.desc = kernels.create_pooling_desc()
        in_edge = self.all_edges[self.input_names[0]]
        n,c,h,w = in_edge.shape
        out_edge = self.all_edges[self.output_names[0]]
        if self.type == "GlobalAveragePool":
            out_edge.shape = [n,c,1,1]
            if self.network_precision == "float32":
                out_edge.dtype = "float32"
                out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
            elif self.network_precision == "float16" :
                out_edge.dtype = "float16"
                out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
            else :
                print("[Error] avgpool infer shape not support!!")
        elif self.type == "MaxPool":
            # floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
            padh = self.params.pads[0]
            padw = self.params.pads[1]
            kh = self.params.kernel_shape[0]
            kw = self.params.kernel_shape[1]
            dilationh = 1
            dilationw = 1
            strideh = self.params.strides[0]
            stridew = self.params.strides[1]
            oh = math.floor((h + padh*2 - ((kh-1)*dilationh + 1))/strideh + 1)
            ow = math.floor((w + padw*2 - ((kw-1)*dilationw + 1))/stridew + 1)
            out_edge.shape = [n,c,oh,ow]
            if self.network_precision == "float32":
                out_edge.dtype = "float32"
                out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
            elif self.network_precision == "float16":
                out_edge.dtype = "float16"
                out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
            else :
                print("[Error] maxpool infer shape not support!!")
        kernels.setup_pooling_descriptor(self.params.kernel_shape, self.params.pads,
                              self.params.strides, in_edge.shape,
                              out_edge.shape, self.type, self.network_precision,
                              "nchw", self.desc)

        print(self.params.kernel_shape)
        print(self.params.pads)
        print(self.params.strides)
        print(in_edge.shape)
        print(out_edge.shape)
        print(self.type)
        print(self.network_precision)
        print(self.desc)

    def infer_layouts(self):
        pass