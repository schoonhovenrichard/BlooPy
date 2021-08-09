import pytest
import random
from hypothesis import given, settings
import hypothesis.strategies as st

import fitness_functions as ff
import dynamic_programming as dp 

@given(st.integers(min_value=2, max_value=6))
@settings(deadline=1000)
def test_dp_hypothesis_mkrandomized(k):
    # Draw parameters M and K for fitness function
    n = 19 // k
    m = n * (k-1)

    mk_func = ff.random_MK_function(m, k)
    mk_func.generate()

    bsol, _ = dp.bruteforce_MK_solve(mk_func)
    dsol = dp.dp_solve_MK(mk_func)
    assert bsol - dsol == 0, "Test failed for DP: bruteforce vs DP (randomized)!"

@given(st.integers(min_value=2, max_value=6))
@settings(deadline=1000)
def test_dp_hypothesis_mkrandomized(k):
    # Draw parameters M and K for fitness function
    n = 19 // k
    m = n * (k-1)

    mk_func = ff.adjacent_MK_function(m, k)
    mk_func.generate()

    bsol, _ = dp.bruteforce_MK_solve(mk_func)
    dsol = dp.dp_solve_MK(mk_func)
    assert bsol - dsol == 0, "Test failed for DP: bruteforce vs DP (adjacent)!"


def dp_test_bruteforce_randomized(tries=10):
    for t in range(tries):
        # Draw parameters M and K for fitness function
        k = random.randint(2,6)
        n = 19 // k
        m = n * (k-1)
        mk_func = ff.random_MK_function(m, k)
        mk_func.generate()

        bsol, _ = dp.bruteforce_MK_solve(mk_func)
        dsol = dp.dp_solve_MK(mk_func)
        assert bsol - dsol == 0, "Test failed for DP: bruteforce vs DP (randomized)!"

def dp_test_bruteforce_adjacent(tries=10):
    for t in range(tries):
        # Draw parameters M and K for fitness function
        k = random.randint(2,6)
        n = 19 // k
        m = n * (k-1)
        mk_func = ff.adjacent_MK_function(m, k)
        mk_func.generate()

        bsol, _ = dp.bruteforce_MK_solve(mk_func)
        dsol = dp.dp_solve_MK(mk_func)
        assert bsol - dsol == 0, "Test failed for DP: bruteforce vs DP (adjacent)!"


if __name__ == '__main__':
    dp_test_bruteforce_adjacent()
    dp_test_bruteforce_randomized()
