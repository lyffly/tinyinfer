from kernels import YTensor, DataType, DataLayout, Dims
import pytest

def test_tensor():
    dims = Dims()
    dims.shape = [64,16,600,600]
    dims.nb_dims = 4
    
    tensor = YTensor()
    tensor.zeros(dims, DataType.float32, DataLayout.nchw)
    tensor.cuda()
    print("gpu ptr: ", tensor.data_ptr)
    tensor.cpu()
    print("cpu ptr: ", tensor.data_ptr)
    tensor.cuda()
    print("gpu ptr: ", tensor.data_ptr)
    tensor.float()
    print("gpu float ptr: ", tensor.data_ptr)
    tensor.half()
    print("gpu half ptr: ", tensor.data_ptr)
    import time
    time.sleep(2)
    print(tensor.shape)
    print(tensor.data_ptr)
    
    assert tensor.shape == dims.shape
    assert tensor.data_ptr > 0 
    
    tensor.free()
    
