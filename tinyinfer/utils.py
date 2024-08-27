import torch
import numpy as np
from kernels import YTensor, DataType, DataLayout, TensorType

def numpy_dtype_2_ytensor_dtype(datatype):
    if datatype == np.float32:
        return DataType.float32
    elif datatype == np.float16:
        return DataType.float16
    elif datatype == np.bool:
        return DataType.bool
    elif datatype == np.int32:
        return DataType.int32
    elif datatype == np.int64:
        return DataType.int64
    elif datatype == np.int8:
        return DataType.int8
    else:
        print("[Error] datatyoe convert wrong: ", datatype)
        return None

def torch_dtype_2_ytensor_dtype(datatype):
    if datatype == torch.float32:
        return DataType.float32
    elif datatype == torch.float16:
        return DataType.float16
    elif datatype == torch.bool:
        return DataType.bool
    elif datatype == torch.int32:
        return DataType.int32
    elif datatype == torch.int64:
        return DataType.int64
    elif datatype == torch.int8:
        return DataType.int8
    else:
        print("[Error] datatyoe convert wrong: ", datatype)
        return None

def data_type_onnx_to_torch(type):
    if type == 0:
        return None
    elif type == 1:
        return torch.float32
    elif type == 2:
        return torch.uint8
    elif type == 3:
        return torch.int8
    elif type == 6:
        return torch.int32
    elif type == 7:
        return torch.int64
    elif type == 9:
        return torch.bool
    elif type == 10:
        return torch.float16
    elif type == 12:
        return torch.uint32
    elif type == 13:
        return torch.uint64
    elif type == 16:
        return torch.bfloat16
    else :
        print("[Error] unknown type : ", type)
        raise TypeError

def get_np_data_ptr(npdata):
    return npdata.__array_interface__['data'][0]
    
def data_type_onnx_to_np(type):
    if type == 0:
        return None
    elif type == 1:
        return np.float32
    elif type == 2:
        return np.uint8
    elif type == 3:
        return np.int8
    elif type == 6:
        return np.int32
    elif type == 7:
        return np.int64
    elif type == 9:
        return np.bool
    elif type == 10:
        return np.float16
    elif type == 12:
        return np.uint32
    elif type == 13:
        return np.uint64
    else :
        print("[Error] unknown type : ", type)
        raise TypeError


def get_gpu_info():
    from cuda import cudart
    _, device_num = cudart.cudaGetDeviceCount()
    if device_num <= 0:
        print("[Error] can not find gpu !!!!")
    else : 
        for i in range(device_num):
            _, prop = cudart.cudaGetDeviceProperties(i)
            print("****"*20)
            print("device index: ", i)
            print("    name: ", str(prop.name))
            print("    total mem (Gb): ", prop.totalGlobalMem/1024/1024/1024)
            print("****"*20)
    
    return device_num
