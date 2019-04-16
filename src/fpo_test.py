import pytest
import fpo as FPO


def test_ap():
    fn1 = lambda v:  v + 1
    fn2 = lambda v:  v * 2
    nums = [1,2,3,4,5]
    assert FPO.ap([fn1, fn2], nums) == [2, 3, 4, 5, 6, 2, 4, 6, 8, 10]

def test_apply():
    def foo(x, y=2): return x + y
    def bar(a, b, c=0): return a + b + c

    f = FPO.apply(fn=foo)
    p = FPO.apply(fn=bar, props=['x','y'])

    assert f({'a': 1, 'b':1}) == 2
    assert f({'x': 3}) == 5
    assert p({'x': 3, 'y': 2}) == 5

# def test_apply():
def test_pluck():
    arr = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
    assert FPO.pluck(arr, 'x', 'y') == [[1, 2], [3, 4]]
    assert FPO.pluck(arr, 'x') == [1, 3]

def test_take():
    items = [2,4,6,8,10]
    assert FPO.take(items, 3) == [2,4,6]
    assert FPO.take(items) == [2]
    assert FPO.take('hello world', 5) == 'hello'