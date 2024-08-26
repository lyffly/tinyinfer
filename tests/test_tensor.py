from kernels import YTensor, DataType, DataLayout, TensorType
import pytest

def test_tensor():
    shape = [64,16,600,600]
    
    tensor = YTensor()
    tensor.zeros(shape, DataType.float32, DataLayout.nchw)
    tensor.tensortype = TensorType.constant
    tensor.cuda()
    print("gpu ptr: ", tensor.data_ptr())
    tensor.cpu()
    print("cpu ptr: ", tensor.data_ptr())
    tensor.cuda()
    print("gpu ptr: ", tensor.data_ptr())
    tensor.half()
    print("gpu ptr: ", tensor.data_ptr())
    tensor.float()
    print("gpu float ptr: ", tensor.data_ptr())
    tensor.half()
    print("gpu half ptr: ", tensor.data_ptr())
    import time
    time.sleep(2)
    print(tensor.shape)
    print(tensor.data_ptr())
    
    assert tensor.data_ptr() > 0 
    
    tensor.free()
    
if __name__ == "__main__" :
    test_tensor()

