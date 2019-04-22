
'''
##FPO.ap(...)
Produces a new list that is a concatenation of sub-lists, each produced by calling FPO.map(..) with each mapper function and the main list.

###Arguments:
    fns:   a list of functions
    list:  a list

###Returns:
    a list of new values

###Example:
    fn1 = lambda increment(v) { return v + 1 }
    def double(v) { return v * 2 }
    nums = [1,2,3,4,5]

###Reference:
    # https://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/
'''
def ap(fns, list):
    matrix = [[fn(v) for v in list] for fn in fns]
    return [n for row in matrix for n in row]



'''
##FPO.apply(...)
Wraps a function to spread out the properties from an object arugment as individual positional arguments

###Arguments:
    fn:     function to wrap
    props:  (optional) list of property names (strings) to indicate the order to spread properties as individual arguments. If omitted, the signature of fn is parsed for its parameter list to try to determine an ordered property list; this detection only works for simple parameters (including those with default parameter value settings).

###Returns:
    function

###Example:
    def foo(x, y=2): return x + y
    def bar(a, b, c=0): return a + b + c

    f = FPO.apply(fn=foo)
    p = FPO.apply(fn=bar, props=['x','y'])

    assert f({'a': 1, 'b':1}) == 2
    assert f({'x': 3}) == 5
    assert p({'x': 3, 'y': 2}) == 5
'''
def apply(fn, props=None):
    def applied(d):
        if props is None:
            return fn(*(d[key] for key,v in d.items()))
        return fn(*(d[key] for key in props))
    return applied



'''
##FPO.binary(...)
Wraps a function to restrict its inputs to dictionary with only two named arguments as specified.

###Arguments:
    fn:     function to wrap
    props:  array of two property names to allow as named arguments
###Returns:
    function
###Example:

###
'''
def binary(fn,props):
    _props = props[slice(0,2)]
    ln = lambda d, props: {key:d[key] for key in props}
    def binaryFn(d):
        return fn(**ln(d, _props))
    return binaryFn



'''
##FPO.complement(...)
Wraps a predicate function -- a function that produces true / false -- to negate its result.
###Arguments:
    fn:     function to wrap
###Returns:
    function
###Example:
    def foo(x,y): return x > y
    def bar(): return True
    f = FPO.complement(foo)
    p = FPO.complement(bar)
    assert foo(3,2) == True
    assert f(3,2) == False
    assert bar() == True
    assert p() == False
'''
def complement(fn):
    def complemented(*arg):
        return False if fn(*arg) is True else True
    return complemented



'''
##FPO.compose(...)
Produces a new function that's the composition of a list of functions. Functions are composed right-to-left (unlike FPO.pipe(..)) from the array.
###Arguments:
    fns:     array of (lambda)functions
###Returns:
    function
###Example:
    f = FPO.compose([
        lambda v: v+2,
        lambda v: v*2,
        lambda v: v-2,
    ])
    assert f(10) == 18
'''
def compose(fns):
    def composed(v):
        result = v
        for fn in reversed(fns):
            result = fn(result)
        return result
    return composed



'''
##FPO.constant(...)
Wraps a value in a function that returns the value.
###Arguments:
    v:     constant value
###Returns:
    function
###Example:
    f = FPO.constant(12)
    assert f() == 12
    assert f(24,9) == 12
    assert f(24) == 12
'''
def constant(v):
    def fn(*arg):
        return v
    return fn



'''
##FPO.curry(...)
Curries a function so that you can pass one argument at a time, each time getting back another function to receive the next argument. Once all arguments are passed, the underlying function is called with the arguments.

Unlike FPO.curryMultiple(..), you can only pass one property argument at a time to each curried function (see example below). If multiple properties are passed to a curried call, only the first property (in enumeration order) will be passed.
###Arguments:
    fn:     function to curry
    n:      number of arguments to curry for
###Returns:
    function
###Example:
    def foo(x,y,z):
        return x + y + z
    f = FPO.curry(fn=foo, n=3)
    v = f(x=1)()(y=2, z=3)(z=4)
    assert v == 7
'''
def curry(fn, n):
    f_args = []
    f_kwargs = {}
    def curried(*args, **kwargs):
        nonlocal f_args, f_kwargs
        if args:
            f_args += args[0]
            if len(f_args) is n:
                return fn(*f_args)
            return curried
        elif kwargs:
            key = list(kwargs)[0]
            f_kwargs[key] = kwargs[key] 
            if len(f_kwargs) is n:
                return fn(**f_kwargs)
            return curried
        else:
            return curried
    return curried



'''
##FPO.curry_multiple(...)
Just like FPO.curry(..), except each curried function allows multiple arguments instead of just one.

Unlike FPO.curryMultiple(..), you can only pass one property argument at a time to each curried function (see example below). If multiple properties are passed to a curried call, only the first property (in enumeration order) will be passed.
###Arguments:
    fn:     function to curry
    n:      number of arguments to curry for
###Returns:
    function
###Example:
    def foo(x,y,z):
        return x + y + z
    f = FPO.curry_multiple(fn=foo, n=3)
    v = f(x=0,y=1)()(x=1)(y=2,z=3)
    assert v == 6
'''
def curry_multiple(fn, n):
    f_args = []
    f_kwargs = {}
    def curried(*args, **kwargs):
        nonlocal f_args, f_kwargs
        if args or kwargs:
            f_args += args
            f_kwargs.update(kwargs)
            if len(f_args) is n or len(f_kwargs) is n:
                return fn(*f_args, **f_kwargs)
            return curried
        else:
            return curried
    return curried



'''
##FPO.filter_in(...)
Commonly known as filter(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new list. Otherwise, the value is omitted.
It is the same as python filter() method
###Arguments:
    fn:     predicate function; called with v (value), i (index), and arr (array) named arguments
    arr:    array to filter against
###Returns:
    list
###Aliases:
    FPO.keep(..)
###Example:
    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_in(fn=is_odd, arr=nums) == [1,3,5]
'''
def filter_in(fn,arr):
    r = []
    for e in arr:
        if fn(e):
           r.append(e)
    return r
keep = filter_in



'''
##FPO.filter_in_dict(...)
Produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new object at the same property name. Otherwise, the value is omitted.
###Arguments:
    fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
    d:      dictionary to filter against
###Returns:
    dictionary
###Aliases:
    FPO.keep_dict(..)
###Example:
    def is_odd(v):
        return v % 2 == 1
    nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
    assert FPO.filter_in_dict(fn=is_odd, d=nums) == {'x':1,'z':3,'l':5}
'''
def filter_in_dict(fn, d):
    r = {}
    for key,v in d.items():
        if fn(v):
            r[key] = v
    return r
keep_dict = filter_in_dict



'''
##FPO.filter_out(...)
The inverse of FPO.filterIn(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new list. Otherwise, the value is included.
###Arguments:
    fn:     predicate function; called with v (value), i (index), and arr (array) named arguments
    arr:    array to filter against
###Returns:
    list
###Aliases:
    FPO.reject(..)
###Example:
    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_out(fn=is_odd, arr=nums) == [2,4]
'''
def filter_out(fn,arr):
    r = []
    for e in arr:
        if fn(e) is not True:
           r.append(e)
    return r
reject = filter_out



'''
##FPO.filter_out_dict(...)
The inverse of FPO.filterInObj(..), produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new object. Otherwise, the value is included at the same property name.
###Arguments:
    fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
    d:      dictionary to filter against
###Returns:
    dictionary
###Aliases:
    FPO.reject_dict(..)
###Example:
    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_in(fn=is_odd, arr=nums) == [1,3,5]
'''
def filter_out_dict(fn, d):
    r = {}
    for key,v in d.items():
        if fn(v) != True:
            r[key] = v
    return r
keep_dict = filter_out_dict



'''
##FPO.pluck(...)
Plucks properties form the given list and return a list of properties' values

###Arguments:
    list:   list
    *args:  properties

###Returns:
    a list of values

###Example:
    arr = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
    assert FPO.pluck(arr, 'x', 'y') == [[1, 2], [3, 4]]
    assert FPO.pluck(arr, 'x') == [1, 3]
'''
# pluck = lambda d, *args: [d[arg] for arg in args]
def pluck(list, *args):
    fn = lambda d, *args: [d[arg] for arg in args]
    r = [fn(o, *args) for o in list]
    if len(args) == 1:
        return [v[0] for v in r]
    else:
        return r



'''
##FPO.take(...)
Returns the specified number of elements from the value, starting from the beginning.

###Arguments:
    iterable:   list/string
    n:          number of elements to take from the beginning of the value; if omitted, defaults to `1`
###Returns:
    list/string
###Example:
    items = [2,4,6,8,10]
    assert FPO.take(items, 3) == [2,4,6]
    assert FPO.take(items) == [2]
    assert FPO.take({'apple','banana','cherry'}, 2) == ['apple','banana']
###
'''
def take(iterable, n=1):
    r = []
    if iterable == None:
        return r
    counter = 0
    for item in iterable:
        if counter == n:
            break
        counter += 1
        r.append(item)
    if isinstance(iterable, str):
        return ''.join(r)
    else:
        return r


