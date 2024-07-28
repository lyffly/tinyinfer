
from kernels import DataLayout
import pytest

def test_data_layput():
    a = DataLayout.NCHW
    print(a)
    b = DataLayout.NHWC
    
    assert a == DataLayout.NCHW
    assert b != a
    assert DataLayout.NHWC ==DataLayout.NHWC

