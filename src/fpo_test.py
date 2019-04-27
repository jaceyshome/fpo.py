import pytest
import fpo as FPO
import random

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

    assert FPO.flat_map_dict(fn=split_evens_in_half, d=nums) == {'a': 1, 'b': 1, 'b_2': 1, 'c': 3, 'd': 2, 'd_2': 2}

def test_flatten():
    nums = [1,2,[3,4],[5,[6,7]]]
    assert FPO.flatten(l=nums) == [1,2,3,4,5,6,7]
    assert FPO.flatten(l=nums,n=1) == [1, 2, 3, 4, 5, [6, 7]]

def test_head():
    assert FPO.head(v={'a':42,'b':56}) == 42
    assert FPO.head(v=[1,2,3,4]) == 1
    assert FPO.head(v=(42,56)) == 42
    assert FPO.head(v='abc') == 'a'
    assert FPO.head(v=[]) == None
    assert FPO.head(v={}) == None
    assert FPO.head(v='') == None

def test_identity():
    FPO.identity( {'v': 42} ) == 42

def test_map_dict():
    def double(v, key): return v * 2
    nums = {'a': 1, 'b': 2, 'c': 3}
    assert FPO.map_dict(fn=double,d=nums) == {'a': 2, 'b': 4, 'c': 6}

def test_map_list():
    def double(v): return v * 2
    nums = [1,2,3]
    assert FPO.map_list(fn=double,l=nums) == [2,4,6]

def test_memoise():
    def sum(x,y):
        return x + y + random.randint(1,101)
    fa = FPO.memoise(fn=sum)
    fb = FPO.memoise(fn=sum, n=1)
    cached_a = fa(2,3)
    assert fa(2,3) == cached_a
    cached_b = fb(2,3)
    assert fb(2,4) == cached_b

def test_n_ary():
    def foo(d): return d
    f = FPO.n_ary(fn=foo, props=['x','y','z'])
    assert f({'x': 1, 'y': 2, 'z': 3, 'w': 4}) == {'x': 1, 'y': 2, 'z': 3}

def test_partial():
    def foo(x,y,z): return x + y + z
    f = FPO.partial(fn=foo, args={'x': 'a'});
    assert f(y='b', z='c') == 'abc'

def test_pick():
    d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.pick(d,props=['x','y']) == {'x': 1, 'y': 2}

def test_pick_all():
    d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.pick_all(d,props=['x','y','r']) == {'x': 1, 'y': 2, 'r': None}

def test_pipe():
    f = FPO.pipe([
        lambda v: v+2,
        lambda v: v*2,
        lambda v: v-2,
    ])
    assert f(10) == 22

def test_pluck():
    l = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
    assert FPO.pluck(l, 'x', 'y') == [[1, 2], [3, 4]]
    assert FPO.pluck(l, 'x') == [1, 3]

def test_prop():
    obj = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.prop(d=obj, prop='y') == 2

def test_reassoc():
    obj = dict(zip(['x','y','z'],[1, 2, 3]))
    assert FPO.reassoc(d=obj, props={'x': 'a', 'y': 'b'}) == {'a': 1, 'b': 2, 'z': 3}
    assert obj == {'x': 1, 'y': 2, 'z': 3}

def test_reduce():
    def str_concat(acc,v):
        return acc + v
    vowels = ["a","e","i","o","u","y"]
    assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
    assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
    assert vowels == ["a","e","i","o","u","y"]

def test_reduce_dict():
    def str_concat(acc,v):
        return acc + v
    vowels = dict(a='a', b='e', c='i', d='o', e='u', f='y')
    assert FPO.reduce_dict(fn=str_concat, d=vowels) == 'aeiouy'
    assert FPO.reduce_dict(fn=str_concat, d=vowels, v='vowels: ') == 'vowels: aeiouy'
    assert vowels == {'a':'a', 'b':'e', 'c':'i', 'd':'o', 'e':'u', 'f':'y'}

def test_reduce_right():
    def str_concat(acc,v):
        return acc + v
    vowels = ["a","e","i","o","u","y"]
    assert FPO.reduce_right(fn=str_concat, l=vowels) == 'yuoiea'
    assert FPO.reduce_right(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: yuoiea'
    assert vowels == ["a","e","i","o","u","y"]

def test_remap():
    def double(x): return x * 2
    def increment(y): return y + 1
    def div3(z): return z / 3
    f = FPO.remap(fn=double, args=dict(v='x'))
    g = FPO.remap(fn=increment, args=dict(v='y'))
    h = FPO.remap(fn=div3, args=dict(v='z'))
    m = FPO.compose(fns=[h,g,f])
    assert f(v=3) == 6
    assert m(v=4) == 3
    assert FPO.map_list(g, [1,4,7,10,13]) == [2,5,8,11,14]
    assert FPO.map_list(m, [1,4,7,10,13]) == [1,3,5,7,9]

def test_set_prop():
    obj = dict(x=1, y=2,z=3)
    assert FPO.set_prop(d=obj, prop='w', v=4) == {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert obj == {'x': 1, 'y': 2, 'z': 3}

def test_tail():
    assert FPO.tail(v={'a':42,'b':56,'c':34}) == {'b':56,'c':34}
    assert FPO.tail(v=[1,2,3,4]) == [2,3,4]
    assert FPO.tail(v=(42,56,32)) == (56,32)
    assert FPO.tail(v='abc') == 'bc'
    assert FPO.tail(v=[]) == None
    assert FPO.tail(v={}) == None
    assert FPO.tail(v='') == None

def test_take():
    items = [2,4,6,8,10]
    assert FPO.take(items, 3) == [2,4,6]
    assert FPO.take(items) == [2]
    assert FPO.take('hello world', 5) == 'hello'

def test_trampoline():
    def sum(total,x):
        if x <= 1:
            return total + x
        return lambda : sum(total+x, x-1)
    assert FPO.trampoline(fn=sum)(0,5) == 15

def test_transducer_filter():
    def is_odd(v):
        return v % 2 == 1
    def list_push(acc, v):
        acc.append(v)
        return acc
    nums = [1,2,3,4,5]
    filter_transducer = FPO.transducer_filter(fn=is_odd)
    r = FPO.transducer_transduce(fn=filter_transducer, co=list_push, v=[], l=nums)
    assert r == [1,3,5]

def test_transducer_transduce():
    def double(v):
        return v * 2
    def is_odd(v):
        return v % 2 == 1
    def list_push(acc, v):
        acc.append(v)
        return acc
    nums = [1,2,3,4,5]
    transducer = FPO.compose(
        fns=[
            FPO.transducer_filter(fn=is_odd),
            FPO.transducer_map(fn=double)
        ]
    )
    result = FPO.transducer_transduce(
        fn=transducer,
        co=list_push,
        v=[],
        l=nums
    )
    assert result == [2,6,10]

def test_transducer_map():
    def double(v):
        return v * 2
    def array_push(acc, v):
        acc.append(v)
        return acc
    nums = [1,2,3,4,5]
    map_transducer = FPO.transducer_map(fn=double)
    r = FPO.transducer_transduce(
        fn=map_transducer,
        co=array_push,
        v=[],
        l=nums
    )
    assert r == [2,4,6,8,10]
    map_reducer = map_transducer(v=array_push)
    assert map_reducer(acc=[], v=3) == [6]
    assert FPO.reduce(fn=map_reducer,v=[],l=nums) == [2,4,6,8,10]

def test_transducer_into():
    def double(v):
        return v * 2
    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    transducer = FPO.compose(
        fns=[
            FPO.transducer_filter(fn=is_odd),
            FPO.transducer_map(fn=double)
        ]
    )
    assert FPO.transducer_into(fn=transducer, v=[], l=nums) == [2,6,10]
    assert FPO.transducer_into(fn=transducer, v=0, l=nums) == 18
    assert FPO.transducer_into(fn=transducer, v='', l=nums) == '2610'

def test_transducer_default():
    assert FPO.transducer_default(acc=3, v=1) == 3

def test_transducer_list():
    arr = [1,2,3]
    FPO.transducer_list(acc=arr,v=4)
    assert arr == [1,2,3,4]

def test_transducer_bool_and():
    assert FPO.transducer_bool_and(acc=True, v=True) == True
    assert FPO.transducer_bool_and(acc=False, v=True) == False

def test_transducer_bool_or():
    assert FPO.transducer_bool_or(acc=True, v=True) == True
    assert FPO.transducer_bool_or(acc=False, v=False) == False
    assert FPO.transducer_bool_or(acc=False, v=True) == True

def test_transducer_number():
    assert FPO.transducer_number( acc=3, v=4) == 7

def test_transducer_string():
    assert FPO.transducer_string( acc='hello', v='world') == 'helloworld'

def test_unapply():
    def foo(x,y):
        return x + y
    f = FPO.unapply(fn=foo, props=['x','y'])
    assert f(1,2) == 3

def test_unary():
    def foo(**kwargs):
        return kwargs
    f = FPO.unary(fn=foo, prop='y')
    assert f(x=1,y=2,z=3) == {'y':2}

def test_uncurry():
    def foo(x,y,z):
        return x + y + z
    f = FPO.curry(fn=foo, n=3)
    p = FPO.uncurry(fn=f)
    assert p(x=1,y=2,z=3) == 6
