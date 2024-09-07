import numpy as np
import torch
from .utils import get_shapes_length, str_dtype_2_ytensor_dtype, str_layout_2_ytensor_layout
from kernels import YTensor, DataType, DataLayout, TensorType

class Edge:
    def __init__(self):
        super().__init__()
        self.name = None
        self.shape = []  # [1,3,8,8]
        self.max_shape = []
        self.layout = "nchw"
        self.dtype = "float16"  # float32 float16 int8 int32 bool
        self.tensor = None # YTensor

    def prepare_data(self):
        pass
    
    def set_shape(self, shape):
        if get_shapes_length(shape) > get_shapes_length(self.max_shape) :
            self.max_shape = shape
        self.shape = shape
        if self.tensor:
            self.tensor.shape = self.max_shape

    def create(self, shape, dtype="float32", layout="nchw") :
        self.set_shape(shape)
        self.dtype = dtype
        self.layout = layout
        
        ytensor = YTensor()
        tensor_dtype = str_dtype_2_ytensor_dtype(self.dtype)
        tensor_layout = str_layout_2_ytensor_layout(self.layout)
        ytensor.zeros(self.max_shape, tensor_dtype, tensor_layout)
        ytensor.tensortype = TensorType.variable
        self.tensor = ytensor
