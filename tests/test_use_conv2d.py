import os
import numpy as np
import sys
sys.path.append("../bin")

import kernels
from cuda import cudart
import pytest
from kernels import YTensor, DataType, DataLayout


def test_conv2d():
    # A + B = C (shape: 2*3*5*5 float)
    _, ptr0 = cudart.cudaMalloc(1 * 3 * 224 * 224 *4) 
    _, ptr1 = cudart.cudaMalloc(64 * 3 * 7 * 7 * 4)
    _, ptr2 = cudart.cudaMalloc(1 * 64 *4)
    _, ptr3 = cudart.cudaMalloc(1 * 64 * 112 * 112 *4)

    in_data0 = np.random.randn(1,3,224,224).astype(np.float32)
    weight_data = np.random.randn(64,3,7,7).astype(np.float32)
    bias_data = np.random.randn(1,64,1,1).astype(np.float32)
    out_data = np.zeros((1,64,112,112),dtype=np.float32)
    _, stream = cudart.cudaStreamCreate()

    cudart.cudaMemcpy(ptr0, in_data0.data, 1 * 3 * 224 * 224 *4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    cudart.cudaMemcpy(ptr1, weight_data.data, 64 * 3 * 7 * 7 * 4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    cudart.cudaMemcpy(ptr2, bias_data.data, 64 *4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    desc = kernels.create_conv2d_desc()
    
    kernel_shape = [7,7]
    pads = [3,3,3,3]
    strides = [2,2]
    dilations = [1,1]
    group = 1
    in_shape = [1,3,224,224]
    w_shape = [64,3,7,7]
    b_shape = [64]
    out_shape = [1,64,112,112]
    network_precision = "float32"
    support_layout = "nchw"
    
    w_tensor = YTensor()
    w_tensor.zeros(w_shape, DataType.float32, DataLayout.nchw)
    w_tensor.copy_numpy_data(weight_data.__array_interface__['data'][0])
    w_tensor.cuda()
    algo = kernels.get_conv2d_algo(
                                kernel_shape, pads, strides,
                                dilations, group, 
                                in_shape, w_shape, b_shape, out_shape, 
                                network_precision, support_layout, stream, desc)

    print("[Python] conv algo find : ", algo)
    workspace_size = kernels.get_conv2d_workspace_size(
                                kernel_shape, pads, strides,
                                dilations, group, 
                                in_shape, w_shape, b_shape, out_shape, 
                                network_precision, support_layout, algo, stream, desc)
    workspace_ptr = 0
    if workspace_size > 0:
        _,  workspace_ptr = cudart.cudaMalloc(workspace_size)

    kernels.conv2d(int(ptr0), w_tensor.data_ptr(), int(ptr2), int(ptr3),
                                workspace_size, int(workspace_ptr), algo,
                                kernel_shape, pads, strides,
                                dilations, group, 
                                in_shape, w_shape, b_shape, out_shape, 
                                network_precision, "nchw", stream, desc)
    
    cudart.cudaMemcpy(out_data.data, ptr3, 1 * 64 * 112 * 112 *4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    print(out_data)
    # print("np.sum(out_data - in_data0 - in_data1) : ", np.abs(np.sum(out_data - in_data0 - in_data1)))

    #assert np.abs(np.sum(out_data - in_data0 - in_data1)) < 0.1

