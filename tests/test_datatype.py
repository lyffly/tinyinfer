
from kernels import DataType
import pytest

def test_data_type():
    a = DataType.INT8
    print(a)
    b = DataType.HALF
    c = DataType.FLOAT16
    
    assert a == DataType.INT8
    assert b == c
    assert DataType.FLOAT32 == DataType.FLOAT32

