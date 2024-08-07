
from kernels import DataType
import pytest

def test_data_type():
    a = DataType.int8
    print(a)
    b = DataType.half
    c = DataType.float16
    
    assert a == DataType.int8
    assert b == c
    assert DataType.float32 == DataType.float32

