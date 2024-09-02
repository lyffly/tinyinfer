import kernels
from cuda import cudart
import pytest
import numpy as np
import torch

def torch_silu(x):
    input = torch.from_numpy(x)
    out = torch.nn.functional.silu(input)
    return out.numpy()


def test_silu_fp32():
    # tensor 2x3x5x5
    _, ptr0 = cudart.cudaMalloc(6 * 25 * 4)
    _, ptr1 = cudart.cudaMalloc(6 * 25 * 4)

    in_data = np.random.randn(2, 3, 5, 5).astype(np.float32)
    out_data = np.zeros((2, 3, 5, 5), dtype=np.float32)
    _, stream = cudart.cudaStreamCreate()

    cudart.cudaMemcpy(
        ptr0, in_data.data, 6 * 25 * 4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )

    kernels.silu(
        int(ptr0),
        int(ptr1),
        [2, 3, 5, 5],
        [2, 3, 5, 5],
        "float32",
        "nchw",
        stream,
    )

    cudart.cudaMemcpy(
        out_data.data, ptr1, 6 * 25 * 4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    
    torch_result = torch_silu(in_data)
    print(
        "np.sum(out_data -  torch_result) : ",
        np.abs(np.sum(out_data - torch_result)),
    )

    assert np.abs(np.sum(out_data - torch_result)) < 0.01