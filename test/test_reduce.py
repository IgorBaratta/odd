import pytest
import odd
import numpy


def test_reduce():
    arr = odd.full(10, 5)
    a = odd.parallel_reduce(arr, numpy.min, axis=None)
    assert a == 5
