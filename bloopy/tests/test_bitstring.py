import pytest
import fitness_functions as ff
from hypothesis import given, settings, assume
import hypothesis.strategies as st
import random


@given(st.integers(min_value=1, max_value=1e6), st.integers(min_value=0))
@settings()
def test_create_bitstring_hypothesis(length, val):
    assume(2**length - 1 >= val)
    bstr = ff.create_bitstring(val, length)
    bstr.reverse()
    transform = int(bstr.to01(),2)
    assert transform == val

def test_create_bitstring(tries=10):
    for t in range(tries):
        length = random.randint(1, 10)
        val = random.randint(0, 2**length -1)
        bstr = ff.create_bitstring(val, length)
        # One runs from left to right, other right to left.
        bstr.reverse()
        transform = int(bstr.to01(),2)
        assert transform == val, "Failed test fitness function, encoding-decoding bitarray"

if __name__ == '__main__':
    test_create_bitstring()
