from kernels import YTensor, DataType, DataLayout, Dims
import pytest

def test_tensor():
    dims = Dims()
    dims.shape = [64,16,600,600]
    dims.nb_dims = 4
    
    tensor = YTensor()
    tensor.zeros(dims, DataType.float32, DataLayout.nchw)
    tensor.cuda()
    tensor.cpu()
    # tensor.free()
    import time
    time.sleep(2)
    print(tensor.shape)
    print(tensor.data_ptr)
    
    assert tensor.shape == dims.shape
    assert tensor.data_ptr > 0 
    
    tensor.free()
    
