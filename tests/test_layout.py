from kernels import DataLayout
import pytest


def test_data_layput():
    a = DataLayout.nchw
    print(a)
    b = DataLayout.nhwc

    assert a == DataLayout.nchw
    assert b != a
    assert DataLayout.nhwc == DataLayout.nhwc
