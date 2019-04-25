import json
import copy

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
    props:  list of two property names to allow as named arguments
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
Produces a new function that's the composition of a list of functions. Functions are composed right-to-left (unlike FPO.pipe(..)) from the list.
###Arguments:
    fns:     list of (lambda)functions
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
            result = fn(v=result)
        return result
    return composed



'''
##FPO.constant(...)
Wraps a value in a fureversed
###Arguments:
    v:     constant vreversed
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
    fn:     predicate function; called with v (value), i (index), and l (list) named arguments
    l:    list to filter against
###Returns:
    list
###Aliases:
    FPO.keep(..)
###Example:
    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_in(fn=is_odd, l=nums) == [1,3,5]
'''
def filter_in(fn,l):
    r = []
    for e in l:
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
    fn:     predicate function; called with v (value), i (index), and l (list) named arguments
    l:    list to filter against
###Returns:
    list
###Aliases:
    FPO.reject(..)
###Example:
    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_out(fn=is_odd, l=nums) == [2,4]
'''
def filter_out(fn,l):
    r = []
    for e in l:
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
    nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
    assert FPO.filter_out_dict(fn=is_odd, d=nums) == {'y':2,'r':4}
'''
def filter_out_dict(fn, d):
    r = {}
    for key,v in d.items():
        if fn(v) != True:
            r[key] = v
    return r
keep_dict = filter_out_dict



'''
##FPO.flat_map(...)
Similar to map(..), produces a new list by calling a mapper function with each value in the original list. If the mapper function returns a list, this list is flattened (one level) into the overall list.
###Arguments:
    fn:  mapper function; called with v (value), i (index), and list(l) named arguments
    l:   list to flat-map against
###Returns:
    list
###Aliases:
    FPO.chain(..)
###Example:
    def split_chars(v): return [*v]
    words = ['hello','world']
    assert split_chars(v=words[0]) == ['h','e','l','l','o']
    assert list(map(split_chars, words)) == [['h','e','l','l','o'],['w','o','r','l','d']]
    assert FPO.flat_map(fn=split_chars, l=words) == ['h','e','l','l','o','w','o','r','l','d']
'''
def flat_map(fn,l):
    t = list(map(fn, l))
    is_all_elements_are_list = True
    for e in t:
        if isinstance(e, list) is not True:
            is_all_elements_are_list = False
    if is_all_elements_are_list is True:
        r = []
        for e in t:
            r += ''.join(e)
        return r
    else:
        return t
chain = flat_map



'''
##FPO.flat_map_dict(...)

###Arguments:
    fn:     mapper function; called with v (value), i (property name), and d (dictionary) named arguments
    d:      dictionary to flat-map against
###Returns:
    dictionary
###Aliases:
    FPO.chain_dict(..)
###Example:
    def split_evens_in_half(v, key):
        if v % 2 == 0:
        return { key: v/2, key+'_2': v/2 }
    return v
    nums = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert split_evens_in_half(v=3, key='c') == 3
    assert split_evens_in_half(v=4, key='d') == {'d':2, 'd_2': 2 }
    assert FPO.map_dict(fn=split_evens_in_half, d=nums) == {'a': 1, 'b': {'b': 1, 'b_2': 1}, 'c': 3, 'd': {'d': 2, 'd_2': 2}}
    assert FPO.flat_map_dict(fn=split_evens_in_half, d=nums) == {'a': 1, 'b': 1, 'b_2': 1, 'c': 3, 'd': 2, 'd_2': 2}
'''
def flat_map_dict(fn,d):
    dd = map_dict(fn,d)
    r = {}
    for key,v in dd.items():
        if isinstance(v, dict) is True:
            r.update(v)
        else:
            r[key] = v
    return r
chain_dict = flat_map_dict



'''
##FPO.flatten(...)
Flattens an array of nested arrays. Optionally, specify how many levels of nesting to flatten out.
###Arguments:
    l:   list to flat-map against
    n:   (optional) the number of levels of nesting to flatten out; if omitted, defaults to Infinity (to flatten any nested depth)
###Returns:
    list
###Example:
    nums = [1,2,[3,4],[5,[6,7]]]
    assert FPO.flatten(l=nums) == [1,2,3,4,5,6,7]
    assert FPO.flatten(l=nums,n=1) == [1, 2, 3, 4, 5, [6, 7]]
'''
def flatten(l, n=-1):
    if n is 0: return l 
    r = []
    for e in l:
        if isinstance(e, list) is True:
            r += flatten(e, n=(n-1))
        else:
            r.append(e)
    return r



'''
##FPO.head(...)
Returns the element as accessed at index 0 of the value.
###Arguments:
    v:   list, tuple, dict, str
###Returns:
    any
###Example:
    nums = [1,2,3,4]
    assert FPO.head(v={'a':42,'b':56}) == 42
    assert FPO.head(v=nums) == 1
    assert FPO.head(v=(42,56)) == 42
    assert FPO.head(v='abc') == 'a'
    assert FPO.head(v=[]) == None
    assert FPO.head(v={}) == None
    assert FPO.head(v='') == None
'''
# https://docs.python.org/2/library/stdtypes.html#truth-value-testing
def head(v):
    if bool(v) is not True:
        return None
    elif isinstance(v, dict) is True:
        return next(iter(v.values()))
    elif isinstance(v, (list, tuple)) is True:
        return v[0]
    elif isinstance(v, str) is True:
        return list(v)[0]



'''
##FPO.identity(...)
Returns the value given to it. Useful as a default placeholder for certain opertaions (i.e., composition, reduction).
###Arguments:
    d:   list
###Returns:
    any
###Example:

###See also: FPO.constant(...)
'''
def identity(d):
    return next(iter(d.values()))



'''
##FPO.map_dict(...)
Produces a new dictionary by calling a mapper function with each property value in the original dictionary. The value the mapper function returns is inserted in the new object at that same property name. The new dictionary will always have the same number of properties as the original dictionary.
###Arguments:
    fn:     mapper function; called with v (value), i (index), and d (dictionary) named arguments
    d:   dictionary to-map against
###Returns:
    dictionary
###Example:
    def double(v, key): return v * 2
    nums = {'a': 1, 'b': 2, 'c': 3}
    assert FPO.map_dict(fn=double,d=nums) == {'a': 2, 'b': 4, 'c': 6}
'''
def map_dict(fn, d):
    r = {}
    for key, v in d.items():
        r[key] = fn(v=v,key=key)
    return r


'''
##FPO.map_list(...)
Produces a new list by calling a mapper function with each value in the original list. The value the mapper function returns is inserted in the new list at that same position. The new list will always be the same length as the original list.
###Arguments:
    fn: mapper function; called with v (value) and l (list) named arguments
    l:  list to map against
###Returns: 
    list
###Example:

'''
def map_list(fn, l):
    r = []
    for v in l:
        r.append(fn(v=v))
    return r



'''
##FPO.memoize(...)
For performance optimization reasons, wraps a function such that it remembers each set of arguments passed to it, associated with that underlying return value. If the wrapped function is called subsequent times with the same set of arguments, the cached return value is returned instead of being recomputed. Each wrapped function instance has its own separate cache, even if wrapping the same original function multiple times.

A set of arguments is "remembered" by being hashed to a string value to use as a cache key. This hashing is done internally with json.dumps(..), which is fast and works with many common value types. However, this hashing is by no means bullet-proof for all types, and does not guarantee collision-free. Use caution: generally, you should only use primitives (number, string, boolean, null, and None) or simple objects (dict, list) as arguments. If you use objects, always make sure to list properties in the same order to ensure proper hashing.

Unary functions (single argument; n of 1) with a primitive argument are the fastest for memoisation, so if possible, try to design functions that way. In these cases, specifying n as 1 will help ensure the best possible performance.

Warning: Be aware that if 1 is initially specified (or detected) for n, additional arguments later passed to the wrapped function are not considered in the memoisation hashing, though they will still be passed to the underlying function as-is. This may cause unexpected results (false-positives on cache hits); always make sure n matches the expected number of arguments.
###Arguments:
    fn: function to wrap
    n:  number of arguments to memoize; if omitted, tries to detect the arity (fn.length) to use.
###Returns:
    list
###Example:
    def sum(x,y):
        return x + y + random.randint(1,101)
    fa = FPO.memoise(fn=sum)
    fb = FPO.memoise(fn=sum, n=1)
    cached_a = fa(2,3)
    assert fa(2,3) == cached_a
    cached_b = fb(2,3)
    assert fb(2,4) == cached_b
'''
def memoise(fn,n=-1):
    cache = {}
    def memoised(*args, **kwargs):
        nonlocal cache
        if bool(args) is True:
            key = json.dumps(take(args, n) if n > 0 else args, sort_keys=True, separators=(',',':'))
        else:
            key = json.dumps(take(kwargs, n) if n > 0 else kwargs, sort_keys=True, separators=(',',':'))
        if key in cache:
            return cache[key]
        else:
            cache[key] = fn(*args, **kwargs)
            return cache[key]
    return memoised



'''
##FPO.n_ary(...)
Wraps a function to restrict its inputs to only the named arguments as specified. It is similar to FPO.pluck.
###Arguments:
    fn:     function to wrap
    props:  list of property names to allow as named arguments; if empty, produces a "nullary" function -- won't receive any arguments.
###Returns:
    function
###Example:
    
'''
def n_ary(fn,props):
    def n_aried(d):
        if bool(props) is not True:
            return fn()
        else:
            r = {}
            for key in props:
                r[key] = d[key]
            return fn(r)
    return n_aried



'''
##FPO.partial(...)
Wraps a function with a new function that already has some of the arguments pre-specified, and is waiting for the rest of them on the next call. Unlike FPO.curry(..), you must specify all the remaining arguments on the next call of the partially-applied function.

With traditional FP libraries, partial(..) works in left-to-right order (as does FPO.std.partial(..)). That's why typically you also need a FPO.std.partialRight(..) if you want to partially-apply from the opposite direction.

However, using named arguments style -- after all, that is the whole point of FPO! -- order doesn't matter. For familiarity sake, FPO.partialRight(..) is provided, but it's just an alias to FPO.partial(..).
###Arguments:
    fn:     function to partially-apply
    args:   object containing the arguments to apply now
###Returns:
    function
###Example:
    def foo(x,y,z): return x + y + z
    f = FPO.partial(fn=foo, args={'x': 'a'});
    assert f(y='b', z='c') == 'abc'
'''
def partial(fn, args):
    def partialed(**kwargs):
        l_kwargs = copy.copy(kwargs)
        l_kwargs.update(args)
        return fn(**l_kwargs)
    return partialed



'''
##FPO.pick(...)
Returns a new dictionary with only the specified properties from the original dictionary. Includes only properties from the original dictionary.
###Arguments:
    d:      dictionary to pick properties from
    props:  list of property names to pick from the object; if a property does not exist on the original dictionary, it is not added to the new dictionary, unlike FPO.pickAll(..).
###Returns:
    dictionary
###Example:
    d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.pick(d,props=['x','y']) == {'x': 1, 'y': 2}
'''
def pick(d,props):
    r = {}
    for i in props:
        if i in d:
            r[i] = d[i]
    return r



'''
##FPO.pick_all(...)
Returns a new dictionary with only the specified properties from the original dictionary. Includes all specified properties.
###Arguments:
    d:      dictionary to pick properties from
    props:  list of property names to pick from the dictionary; even if a property does not exist on the original dictionary, it is still added to the new object with an undefined value, unlike FPO.pick(..).
###Returns:
    dictionary
###Example:
    d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.pick_all(d,props=['x','y','r']) == {'x': 1, 'y': 2, 'r': None}
'''
def pick_all(d, props):
    r = {}
    for i in props:
        if i in d:
            r[i] = d[i]
        else:
            r[i] = None
    return r



'''
##FPO.pipe(...)
Produces a new function that's the composition of a list of functions. Functions are composed left-to-right (unlike FPO.compose(..)) from the array.
###Arguments:
    fns:    list of funcitons
###Returns:
    function
###Example:
    f = FPO.pipe([
        lambda v: v+2,
        lambda v: v*2,
        lambda v: v-2,
    ])
    assert f(10) == 22
'''
def pipe(fns):
    def piped(v):
        result = v
        for fn in fns:
            result = fn(v=result)
        return result
    return piped



'''
##FPO.pluck(...)
Plucks properties form the given list and return a list of properties' values
###Arguments:
    l:   list
    *args:  properties
###Returns:
    a list of values
###Example:
    l = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
    assert FPO.pluck(l, 'x', 'y') == [[1, 2], [3, 4]]
    assert FPO.pluck(l, 'x') == [1, 3]
'''
def pluck(l, *args):
    fn = lambda d, *args: [d[arg] for arg in args]
    r = [fn(o, *args) for o in l]
    if len(args) == 1:
        return [v[0] for v in r]
    else:
        return r



'''
##FPO.prop(...)
Extracts a property's value from a dictionary.
###Arguments:
    d:    dictionary to pull the property value from
    prop: property name to pull from the dictionary
###Returns:
    any
###Example:
    obj = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.prop(d=obj, prop='y') == 2
'''
def prop(d,prop):
    return d[prop]
    


'''
##FPO.reassoc(...)
Like a mixture between FPO.pick(..) and FPO.setProp(..), creates a new dictionary that has properties remapped from original names to new names. Any properties present on the original dictionary that aren't remapped are copied with the same name.
###Arguments:
    d:      dictionary to remap properties from
    props:  dictionary whose key/value pairs are sourceProp: targetProp remappings
###Returns:
    dictionary
###Example:
    obj = dict(zip(['x','y','z'],[1, 2, 3]))
    assert FPO.reassoc(d=obj, props={'x': 'a', 'y': 'b'}) == {'a': 1, 'b': 2, 'z': 3}
    assert obj == {'x': 1, 'y': 2, 'z': 3}
'''
def reassoc(d,props):
    r = {}
    for k,v in d.items():
        if k in props:
            r[props[k]] = d[k]
        else:
            r[k] = d[k]
    return r



'''
##FPO.reduce(..)
Processes a list from left-to-right (unlike FPO.reduceRight(..)), successively combining (aka "reducing", "folding") two values into one, until the entire list has been reduced to a single value. An initial value for the reduction can optionally be provided.
###Arguments:
    fn: reducer function; called with acc (accumulator), v (value) and l (list) named arguments
    l:  list to reduce
    v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
###Returns: 
    any
###Example:
    def str_concat(acc,v):
        return acc + v
    vowels = ["a","e","i","o","u","y"]
    assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
    assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
    assert vowels == ["a","e","i","o","u","y"]
'''
def reduce(fn,l,v=None):
    r = l[0]
    for e in l[1:]:
        r = fn(acc=r, v=e)
    if bool(v) is True:
        return v + r
    return r



'''
##FPO.reduce_dict(..)
Processes an dictionary's properties (in enumeration order), successively combining (aka "reducing", "folding") two values into one, until all the dictionary's properties have been reduced to a single value. An initial value for the reduction can optionally be provided.
###Arguments:
    fn: reducer function; called with acc (accumulator), v (value) and l (list) named arguments
    d:  dictionary to reduce
    v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
###Returns: 
    any
###Example:
    def str_concat(acc,v):
        return acc + v
    vowels = ["a","e","i","o","u","y"]
    assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
    assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
    assert vowels == ["a","e","i","o","u","y"]
'''
def reduce_dict(fn,d,v=None):
    init_k = next(iter(d))
    r = d[init_k]
    for key,value in d.items():
        if key is not init_k:
            r = fn(acc=r, v=value)
    if bool(v) is True:
        return v + r
    return r



'''
##FPO.reduce_right(..)
Processes a list from right-to-left (unlike FPO.reduce(..)), successively combining (aka "reducing", "folding") two values into one, until the entire list has been reduced to a single value.
An initial value for the reduction can optionally be provided. If the array is empty, the initial value is returned (or undefined if it was omitted).
###Arguments:
    fn: reducer function; called with acc (accumulator), v (value) and l (list) named arguments
    l:  list to reduce
    v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
###Returns: 
    any
###Example:
    def str_concat(acc,v):
        return acc + v
    vowels = ["a","e","i","o","u","y"]
    assert FPO.reduce_right(fn=str_concat, l=vowels) == 'yuoiea'
    assert FPO.reduce_right(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: yuoiea'
    assert vowels == ["a","e","i","o","u","y"]
'''
def reduce_right(fn,l,v=None):
    rl = l[::-1]
    r = rl[0]
    for e in rl[1:]:
        r = fn(acc=r, v=e)
    if bool(v) is True:
        return v + r
    return r



'''
##FPO.remap(..)
Remaps the expected named arguments of a function. This is useful to adapt a function to be used if the arguments passed in will be different than what the function expects.
A common usecase will be to adapt a function so it's suitable for use as a mapper/predicate/reducer function, or for composition.
###Arguments:
    fn:     function to remap
    args:   dictionary whose key/value pairs represent the origArgName: newArgName mappings
###Returns: 
    function
###Example:
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
'''
def remap(fn, args):
    def remaped(**kwargs):
        print('AAAA', kwargs)
        l_kwargs = reassoc(kwargs,props=args)
        return fn(**l_kwargs)
    return remaped



'''
##FPO.set_prop(...)
Creates a shallow clone of a dictionary, assigning the specified property value to the new dictionary.
###Arguments:
    d:      (optional) object to clone; if omitted, defaults to a new empty dictionary
    prop:   property name where to set the value on the new dictionary
    v:      value
###Returns:
    any
###Example:
    obj = dict(x=1, y=2,z=3)
    assert FPO.set_prop(d=obj, prop='w', v=4) == {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert obj == {'x': 1, 'y': 2, 'z': 3}
'''
def set_prop(d,prop,v):
    if bool(d) is True:
        r = copy.copy(d)
    else:
        r = {}
    r[prop] = v
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


