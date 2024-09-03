import kernels
from cuda import cudart
import pytest
import numpy as np
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) # gamma parameter
    
    def _norm(self, x: torch.tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) # (B, seq_len, dim)

    def forward(self, x: torch.tensor):
        return self.weight * self._norm(x.float()).type_as(x) # (dim) * (B, seq_len, dim) --> (B, seq_len, dim)


def torch_rms_norm(x):
    input = torch.from_numpy(x)
    model = RMSNorm(512)
    out = model(input)
    return out.detach().numpy()


def test_rms_norm_fp32():
    # tensor batch
    batch = 2
    seq_len = 5
    dim = 512
    _, ptr0 = cudart.cudaMalloc(batch * seq_len *dim * 4)
    _, w_ptr = cudart.cudaMalloc(dim * 4)
    _, ptr1 = cudart.cudaMalloc(batch * seq_len *dim * 4)

    in_data = np.random.randn(batch , seq_len , dim).astype(np.float32)
    w_data = np.ones(dim).astype(np.float32)
    out_data = np.zeros((batch , seq_len , dim), dtype=np.float32)
    _, stream = cudart.cudaStreamCreate()

    cudart.cudaMemcpy(
        ptr0, in_data.data, batch * seq_len *dim * 4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )
    cudart.cudaMemcpy(
        w_ptr, w_data.data, dim * 4, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    )

    kernels.rms_norm(
        int(ptr0),
        int(w_ptr),
        int(ptr1),
        [batch , seq_len , dim],
        [batch , seq_len , dim],
        1e-6,
        "float32",
        stream,
    )

    cudart.cudaMemcpy(
        out_data.data, ptr1, batch * seq_len *dim * 4, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    )
    
    torch_result = torch_rms_norm(in_data)
    print(
        "np.sum(out_data -  torch_result) : ",
        np.abs(np.sum(out_data - torch_result)),
    )
    print(out_data)
    print(torch_result)

    assert np.abs(np.sum(out_data - torch_result)) < 0.01