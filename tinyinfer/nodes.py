from .params import *
import math
import torch
import numpy as np
import torch.nn.functional as F
from cuda import cudart

class Node:
    def __init__(self):
        self.name = None
        self.type = None
        self.params = None
        self.input_names = None
        self.output_names = None
        self.input_shapes = None
        self.output_shapes = None
        self.input_dtypes = None
        self.output_dtypes = None
        self.input_layouts = None
        self.output_layouts = None
        self.all_edges = None
        self.network_precision = "float32"

    def bind_all_edges(self, edges):
        self.all_edges = edges

    def run(self, stream):
        print("node run")
        pass

    def print(self):
        print("\nnode name:", self.name)
        print("     type:", self.type)


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

    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        w_edge = self.all_edges[self.input_names[1]]
        b_edge = None
        if len(self.input_names) > 2:
            b_edge = self.all_edges[self.input_names[2]]
        out_edge = self.all_edges[self.output_names[0]]
        try:
            # print("\n****use cudnn conv")
            import kernels
            if self.algo < 0:
                self.algo = kernels.get_conv2d_algo(
                                self.params.kernel_shape, self.params.pads, self.params.strides,
                                self.params.dilations, self.params.group, 
                                in_edge.shape, w_edge.shape, b_edge.shape, out_edge.shape, 
                                self.network_precision, "nchw")
                assert self.algo >= 0
                # print("conv algo : ", self.algo)
                self.workspace_size = kernels.get_conv2d_workspace_size(
                                self.params.kernel_shape, self.params.pads, self.params.strides,
                                self.params.dilations, self.params.group, 
                                in_edge.shape, w_edge.shape, b_edge.shape, out_edge.shape, 
                                self.network_precision, "nchw", self.algo)
                assert self.workspace_size >= 0
                _, self.workspace_ptr = cudart.cudaMalloc(self.workspace_size)
                # print("conv workspace size : ", self.workspace_size)

            kernels.conv2d(in_edge.tensor.data_ptr(), w_edge.tensor.data_ptr(), 
                            b_edge.tensor.data_ptr(), out_edge.tensor.data_ptr(), 
                            self.workspace_size, self.workspace_ptr, self.algo,
                            self.params.kernel_shape, self.params.pads, self.params.strides,
                            self.params.dilations, self.params.group, 
                            in_edge.shape, w_edge.shape, b_edge.shape, out_edge.shape, 
                            self.network_precision, "nchw", stream)
            # print(out_edge.tensor[0][0][0][:10])
        except:
        # if True:
            # print("****use pytorch conv2d")
            out_edge.tensor = F.conv2d(in_edge.tensor, w_edge.tensor, b_edge.tensor, stride=self.params.strides,
                    padding=self.params.pads[:2], groups=self.params.group)
            # print(out_edge.tensor[0][0][0][:10])
    
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        weights_edge = self.all_edges[self.input_names[1]]
        if len(self.input_names) > 2:
            bias_edge = self.all_edges[self.input_names[2]]
        n,c,h,w = in_edge.shape
        oc,c,kh,kw = weights_edge.shape
        self.inc = c
        self.outc = oc
        padh = (self.params.pads[0] +self.params.pads[2] ) /2
        padw = (self.params.pads[1] +self.params.pads[3] ) /2
        oh = math.floor((h + 2*padh - self.params.dilations[0] * (kh -1) -1)/self.params.strides[0]  +1)
        ow = math.floor((h + 2*padw - self.params.dilations[1] * (kw -1) -1)/self.params.strides[1]  +1)
        
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = [n,oc,oh,ow]
        if self.network_precision == "float32" :
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.network_precision == "float16" :
            out_edge.dtype = "float16"
            weights_edge.tensor = weights_edge.tensor.half()
            if bias_edge:
                bias_edge.tensor = bias_edge.tensor.half()
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] conv infer shape not support!!")
    
    def infer_layouts(self):
        pass


class ActivationNode(Node):
    def __init__(self):
        super().__init__()
        self.params = ActivationParams()
        self.type = "Activation"
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        
        try:
            import kernels
            #print("****use cuda activation")
            kernels.activation(in_edge.tensor.data_ptr(), out_edge.tensor.data_ptr(), self.params.alpha,
                                self.params.beta, in_edge.shape, out_edge.shape, self.network_precision,
                                "nchw", self.type, stream)
        except:
            #print("****use pytorch activation")
            out_edge.tensor = torch.max(torch.tensor(0), in_edge.tensor)
        
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        n,c,h,w = in_edge.shape

        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = in_edge.shape
        if self.network_precision == "float32" :
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.network_precision == "float16" :
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] activation infer shape not support!!")
    
    def infer_layouts(self):
        pass


class ElementwiseNode(Node):
    def __init__(self):
        super().__init__()
        self.params = ElementwiseParams()
        self.type = "Elementwise"
    
    def run(self, stream):
        in_edge0 = self.all_edges[self.input_names[0]]
        in_edge1 = self.all_edges[self.input_names[1]]
        out_edge = self.all_edges[self.output_names[0]]
        try:
            import kernels
            # print("****use cuda elementwise")
            kernels.elementwise(in_edge0.tensor.data_ptr(), in_edge1.tensor.data_ptr(), out_edge.tensor.data_ptr(),
                                in_edge0.shape, in_edge1.shape, out_edge.shape, self.network_precision,
                                "nchw", self.type, stream)
        except:
            # print("****use pytorch elementwise")
            if self.type == "Add":
                out_edge.tensor = in_edge0.tensor + in_edge1.tensor
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        n,c,h,w = in_edge.shape

        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = in_edge.shape
        if self.network_precision == "float32" :
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.network_precision == "float16" :
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] elementwise infer shape not support!!")
    
    def infer_layouts(self):
        pass


class PoolNode(Node):
    def __init__(self):
        super().__init__()
        self.params = PoolParams()
        self.type = "pool"

    def run(self, stream):
        for name in self.input_names:
            edge = self.all_edges[name]
        for name in self.output_names:
            edge = self.all_edges[name]
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        if self.type == "GlobalAveragePool":
            n,c,h,w = in_edge.tensor.shape
            out_edge.tensor = F.avg_pool2d(in_edge.tensor,(h,w))
        elif self.type == "MaxPool":
            out_edge.tensor = F.max_pool2d(in_edge.tensor, self.params.kernel_shape,
                                        stride=self.params.strides,padding=self.params.pads[:2])


    def infer_shapes(self):
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
            oh = math.floor((h + padh - ((kh-1)*dilationh + 1))/strideh + 1)
            ow = math.floor((w + padw - ((kw-1)*dilationw + 1))/stridew + 1)
            out_edge.shape = [n,c,oh,ow]
            if self.network_precision == "float32":
                out_edge.dtype = "float32"
                out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
            elif self.network_precision == "float16":
                out_edge.dtype = "float16"
                out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
            else :
                print("[Error] maxpool infer shape not support!!")


    def infer_layouts(self):
        pass


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
        
        try: # use cuda cublas
            import kernels
            if self.workspace_size : 
                _, self.workspace_ptr = cudart.cudaMalloc(self.workspace_size)
            if bias_edge:
                kernels.gemm(in_edge.tensor.data_ptr(), w_edge.tensor.data_ptr(), bias_edge.tensor.data_ptr(), out_edge.tensor.data_ptr(),
                        self.workspace_size, self.workspace_ptr, self.params.alpha, self.params.beta, 
                        self.params.transA, self.params.transB, in_edge.shape, w_edge.shape, bias_edge.shape,
                        out_edge.shape, self.network_precision, stream)
            else :
                kernels.gemm(in_edge.tensor.data_ptr(), w_edge.tensor.data_ptr(), 0, out_edge.tensor.data_ptr(),
                        self.workspace_size, self.workspace_ptr, self.params.alpha, self.params.beta, 
                        self.params.transA, self.params.transB, in_edge.shape, w_edge.shape, [], out_edge.shape, 
                        self.network_precision, stream)
            # print("****use cublas gemm\n")
        except: # use torch
            if self.params.transB:
                out_edge.tensor = torch.matmul(in_edge.tensor, w_edge.tensor.T)
            else:
                out_edge.tensor = torch.matmul(in_edge.tensor, w_edge)
            # print("****use pytorch gemm\n")
    
    def __del__(self):
        if self.workspace_ptr:
            try:
                cudart.cudaFree(self.workspace_ptr)
            except:
                pass
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        weights_edge = self.all_edges[self.input_names[1]]
        if len(self.input_names) > 2:
            bias_edge = self.all_edges[self.input_names[2]]
        m, k = in_edge.shape
        n = 0
        out_edge = self.all_edges[self.output_names[0]]
        if self.params.transB == 1: 
            n, _ = weights_edge.shape
        else :
            _, n = weights_edge.shape
        
        out_edge.shape = [m, n]
        if self.network_precision == "float32" :
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.network_precision == "float16" :
            out_edge.dtype = "float16"
            weights_edge.tensor = weights_edge.tensor.half()
            if bias_edge:
                bias_edge.tensor = bias_edge.tensor.half()
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] gemm infer shape not support!!")

    def infer_layouts(self):
        pass


class FlattenNode(Node):
    def __init__(self):
        super().__init__()
        self.params = FlattenParams()
        self.type = "Flatten"
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.tensor = in_edge.tensor.reshape(out_edge.shape)
    
    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        n,c,h,w = in_edge.shape
        out_edge = self.all_edges[self.output_names[0]]
        if self.params.axis == 1 and self.network_precision == "float32" :
            out_edge.shape = [n, c*h*w]
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.params.axis == 1 and self.network_precision == "float16" :
            out_edge.shape = [n, c*h*w]
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] flatten infer shape not support!!")
    
    def infer_layouts(self):
        pass


class CastNode(Node):
    def __init__(self, in_dtype, out_dtype):
        super().__init__()
        self.params = CastParams()
        self.type = "Cast"
        self.in_dtype = in_dtype    #"float32"
        self.out_dtype = out_dtype  #"float16"
    
    def run(self, stream):
        in_edge = self.all_edges[self.input_names[0]]
        out_edge = self.all_edges[self.output_names[0]]
        
        try: # use cuda cublas
            import kernels
            kernels.cast(in_edge.tensor.data_ptr(), out_edge.tensor.data_ptr(),
                        in_edge.shape, out_edge.shape, "nchw", self.in_dtype, self.out_dtype, stream)
            #print("****use cuda cast\n")
        except:
            if self.out_dtype == "float32":
                out_edge.tensor = in_edge.tensor.float()
            elif self.out_dtype == "float16":
                out_edge.tensor = in_edge.tensor.half()
            #print("****use pytorch cast\n")

    def infer_shapes(self):
        in_edge = self.all_edges[self.input_names[0]]
        in_edge.dtype = self.in_dtype
        out_edge = self.all_edges[self.output_names[0]]
        out_edge.shape = in_edge.shape
        if self.out_dtype == "float32":
            out_edge.dtype = "float32"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float32, requires_grad=False)
        elif self.out_dtype == "float16":
            out_edge.dtype = "float16"
            out_edge.tensor = torch.zeros(out_edge.shape, dtype=torch.float16, requires_grad=False)
        else :
            print("[Error] cast infer shape not support!!")

    def infer_layouts(self):
        pass