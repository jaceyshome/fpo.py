import pytest
import fpo as FPO

def test_pluck():
    arr = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
    assert FPO.pluck(arr, 'x', 'y') == [[1, 2], [3, 4]]
    assert FPO.pluck(arr, 'x') == [1, 3]

def test_ap():
    fn1 = lambda v:  v + 1
    fn2 = lambda v:  v * 2
    nums = [1,2,3,4,5]
    assert FPO.ap([fn1, fn2], nums) == [2, 3, 4, 5, 6, 2, 4, 6, 8, 10]