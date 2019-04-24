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

def test_binary():
    def foo(x,y): return x + y
    def bar(a,b,c=1): return a + b + c
    f = FPO.binary(fn=foo, props=['x','y'])
    p = FPO.binary(fn=bar, props=['a','b'])
    assert f({'x':1, 'y':2, 'z':4}) == 3
    assert p({'a':2,'b':4,'c':6}) == 7

def test_complement():
    def foo(x,y): return x > y
    def bar(): return True
    f = FPO.complement(foo)
    p = FPO.complement(bar)
    assert foo(3,2) == True
    assert f(3,2) == False
    assert bar() == True
    assert p() == False

def test_compose():
    f = FPO.compose([
        lambda v: v+2,
        lambda v: v*2,
        lambda v: v-2,
    ])
    assert f(10) == 18

def test_constant():
    f = FPO.constant(12)
    assert f() == 12
    assert f(24,9) == 12
    assert f(24) == 12

def test_curry():
    def foo(x,y,z):
        return x + y + z
    f = FPO.curry(fn=foo, n=3)
    v = f(x=1)()(y=2, z=3)(z=4)
    assert v == 7

def test_curry_multiple():
    def foo(x,y,z):
        return x + y + z
    f = FPO.curry_multiple(fn=foo, n=3)
    v = f(x=0,y=1)()(x=1)(y=2,z=3)
    assert v == 6

def test_filter_in():
    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_in(fn=is_odd, l=nums) == [1,3,5]

def test_filter_dict_in():
    def is_odd(v):
        return v % 2 == 1
    nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
    assert FPO.filter_in_dict(fn=is_odd, d=nums) == {'x':1,'z':3,'l':5}

def test_filter_out():
    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_out(fn=is_odd, l=nums) == [2,4]

def test_filter_out_dict():
    def is_odd(v):
        return v % 2 == 1
    nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
    assert FPO.filter_out_dict(fn=is_odd, d=nums) == {'y':2,'r':4}

def test_flat_map():
    def split_chars(v): return [*v]
    words = ['hello','world']
    assert split_chars(v=words[0]) == ['h','e','l','l','o']
    assert list(map(split_chars, words)) == [['h','e','l','l','o'],['w','o','r','l','d']]
    assert FPO.flat_map(fn=split_chars, l=words) == ['h','e','l','l','o','w','o','r','l','d']

def test_flat_map_dict():
    def split_evens_in_half(v, key):
        if v % 2 == 0:
            return { key: v/2, key+'_2': v/2 }
        return v
    nums = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert split_evens_in_half(v=3, key='c') == 3
    assert split_evens_in_half(v=4, key='d') == {'d':2, 'd_2': 2 }
    assert FPO.map_dict(fn=split_evens_in_half, d=nums) == {'a': 1, 'b': {'b': 1, 'b_2': 1}, 'c': 3, 'd': {'d': 2, 'd_2': 2}}

def test_map_dict():
    def double(v, key): return v * 2
    nums = {'a': 1, 'b': 2, 'c': 3}
    assert FPO.map_dict(fn=double,d=nums) == {'a': 2, 'b': 4, 'c': 6}

def test_pluck():
    l = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
    assert FPO.pluck(l, 'x', 'y') == [[1, 2], [3, 4]]
    assert FPO.pluck(l, 'x') == [1, 3]

def test_take():
    items = [2,4,6,8,10]
    assert FPO.take(items, 3) == [2,4,6]
    assert FPO.take(items) == [2]
    assert FPO.take('hello world', 5) == 'hello'
