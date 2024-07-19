import os
import numpy as np
import sys
sys.path.append("../bin")

import kernels
from cuda import cudart
import pytest

# A + B = C (shape: 2*3*5*5 float)
_, ptr0 = cudart.cudaMalloc(6 * 25 * 4)
_, ptr1 = cudart.cudaMalloc(6 * 25 * 4)
_, ptr2 = cudart.cudaMalloc(6 * 25 * 4)

in_data0 = np.random.randn(2,3,5,5).astype(np.float32)
in_data1 = np.random.randn(2,3,5,5).astype(np.float32)
out_data = np.zeros((2,3,5,5),dtype=np.float32)

cudart.cudaMemcpy(ptr0, in_data0.data, 6 * 25 * 4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
cudart.cudaMemcpy(ptr1, in_data1.data, 6 * 25 * 4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

kernels.elementwise(int(ptr0), int(ptr1), int(ptr2), [2,3,5,5], [2,3,5,5], [2,3,5,5], "float32", "nchw")

cudart.cudaMemcpy(out_data.data, ptr2, 6 * 25 * 4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
print("np.sum(out_data - in_data0 - in_data1) : ")
print(np.sum(out_data - in_data0 - in_data1))

