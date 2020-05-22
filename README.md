#FPO.py

Python Clone of FPOjs https://github.com/getify/FPO

## Prerequisites
Python 3.6.x

### Install pip3 and major libraries
https://linoxide.com/linux-how-to/install-flask-python-ubuntu/

```
sudo apt-get update
sudo apt-get install python3-pip python3-dev
sudo apt-get install python3.6-venv
```
### Setup

Create a virtual environment to store this project requirements
```
python3 -m venv env
```

This will install a local copy of Python and pip into a directory called flasksupportserverenv within your project directory.
Before we install applications within the virtual environment, we need to activate it. You can do so by typing:

```
source env/bin/activate
```

###Install dependencies
```
pip3 install -r requirements.txt
```

### Running the virtual environment
```
source env/bin/activate
```

### Doc
https://pdoc3.github.io/pdoc/doc/pdoc/

Update doc
```
pdoc --html src/fpo.py --overwrite
```
### Running the test

```
ptw # pytest watch mode
pytest -v run pytest
```

<main>

<article id="content">

<header>

# `fpo` module

</header>

<section id="section-intro"><details class="source"><summary>Source code</summary>

    import json
    import copy

    def ap(fns, list):
        '''
        ## FPO.ap(...)
        Produces a new list that is a concatenation of sub-lists, each produced by calling FPO.map(..) with each mapper function and the main list.
        ### Arguments:
            fns:   a list of functions
            list:  a list
        ### Returns:
            a list of new values
        ### Example:
            fn1 = lambda increment(v) { return v + 1 }
            def double(v) { return v * 2 }
            nums = [1,2,3,4,5]
        ### Reference:
            # https://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/
        '''
        matrix = [[fn(v) for v in list] for fn in fns]
        return [n for row in matrix for n in row]

    def apply(fn, props=None):
        '''
        ## FPO.apply(...)
        Wraps a function to spread out the properties from an object arugment as individual positional arguments
        ### Arguments:
            fn:     function to wrap
            props:  (optional) list of property names (strings) to indicate the order to spread properties as individual arguments. If omitted, the signature of fn is parsed for its parameter list to try to determine an ordered property list; this detection only works for simple parameters (including those with default parameter value settings).
        ### Returns:
            function
        ### Example:
            def foo(x, y=2): return x + y
            def bar(a, b, c=0): return a + b + c
            f = FPO.apply(fn=foo)
            p = FPO.apply(fn=bar, props=['x','y'])
            assert f({'a': 1, 'b':1}) == 2
            assert f({'x': 3}) == 5
            assert p({'x': 3, 'y': 2}) == 5
        '''
        def applied(d):
            if props is None:
                return fn(*(d[key] for key,v in d.items()))
            return fn(*(d[key] for key in props))
        return applied

    def binary(fn,props):
        '''
        ## FPO.binary(...)
        Wraps a function to restrict its inputs to dictionary with only two named arguments as specified.

        ### Arguments:
            fn:     function to wrap
            props:  list of two property names to allow as named arguments
        ### Returns:
            function
        ### Example:
            def foo(x,y): return x + y
            def bar(a,b,c=1): return a + b + c
            f = FPO.binary(fn=foo, props=['x','y'])
            p = FPO.binary(fn=bar, props=['a','b'])
            assert f({'x':1, 'y':2, 'z':4}) == 3
            assert p({'a':2,'b':4,'c':6}) == 7
        '''
        _props = props[slice(0,2)]
        ln = lambda d, props: {key:d[key] for key in props}
        def binaryFn(d):
            return fn(**ln(d, _props))
        return binaryFn

    def complement(fn):
        '''
        ## FPO.complement(...)
        Wraps a predicate function -- a function that produces true / false -- to negate its result.
        ### Arguments:
            fn:     function to wrap
        ### Returns:
            function
        ### Example:
            def foo(x,y): return x > y
            def bar(): return True
            f = FPO.complement(foo)
            p = FPO.complement(bar)
            assert foo(3,2) == True
            assert f(3,2) == False
            assert bar() == True
            assert p() == False
        '''
        def complemented(*arg):
            return False if fn(*arg) is True else True
        return complemented

    def compose(fns):
        '''
        ## FPO.compose(...)
        Produces a new function that's the composition of a list of functions. Functions are composed right-to-left (unlike FPO.pipe(..)) from the list.
        ### Arguments:
            fns:     list of (lambda)functions
        ### Returns:
            function
        ### Example:
            f = FPO.compose([
                lambda v: v+2,
                lambda v: v*2,
                lambda v: v-2,
            ])
            assert f(10) == 18
        '''
        def composed(v):
            result = v
            for fn in reversed(fns):
                result = fn(v=result)
            return result
        return composed

    def constant(v):
        '''
        ## FPO.constant(...)
        Wraps a value in a fureversed
        ### Arguments:
            v:     constant vreversed
        ### Returns:
            function
        ### Example:
            f = FPO.constant(12)
            assert f() == 12
            assert f(24,9) == 12
            assert f(24) == 12
        '''
        def fn(*arg):
            return v
        return fn

    def curry(fn, n):
        '''
        ## FPO.curry(...)
        Curries a function so that you can pass one argument at a time, each time getting back another function to receive the next argument. Once all arguments are passed, the underlying function is called with the arguments.

        Unlike FPO.curryMultiple(..), you can only pass one property argument at a time to each curried function (see example below). If multiple properties are passed to a curried call, only the first property (in enumeration order) will be passed.
        ### Arguments:
            fn:     function to curry
            n:      number of arguments to curry for
        ### Returns:
            function
        ### Example:
            def foo(x,y,z):
                return x + y + z
            f = FPO.curry(fn=foo, n=3)
            v = f(x=1)()(y=2, z=3)(z=4)
            assert v == 7
        '''
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

    def curry_multiple(fn, n):
        '''
        ## FPO.curry_multiple(...)
        Just like FPO.curry(..), except each curried function allows multiple arguments instead of just one.

        Unlike FPO.curryMultiple(..), you can only pass one property argument at a time to each curried function (see example below). If multiple properties are passed to a curried call, only the first property (in enumeration order) will be passed.
        ### Arguments:
            fn:     function to curry
            n:      number of arguments to curry for
        ### Returns:
            function
        ### Example:
            def foo(x,y,z):
                return x + y + z
            f = FPO.curry_multiple(fn=foo, n=3)
            v = f(x=0,y=1)()(x=1)(y=2,z=3)
            assert v == 6
        '''
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

    def filter_in(fn,l):
        '''
        ## FPO.filter_in(...)
        Commonly known as filter(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new list. Otherwise, the value is omitted.
        It is the same as python filter() method
        ### Arguments:
            fn:     predicate function; called with v (value), i (index), and l (list) named arguments
            l:    list to filter against
        ### Returns:
            list
        ### Aliases:
            FPO.keep(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = [1,2,3,4,5]
            assert FPO.filter_in(fn=is_odd, l=nums) == [1,3,5]
        '''
        r = []
        for e in l:
            if fn(e):
               r.append(e)
        return r
    keep = filter_in

    def filter_in_dict(fn, d):
        '''
        ## FPO.filter_in_dict(...)
        Produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new object at the same property name. Otherwise, the value is omitted.
        ### Arguments:
            fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
            d:      dictionary to filter against
        ### Returns:
            dictionary
        ### Aliases:
            FPO.keep_dict(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
            assert FPO.filter_in_dict(fn=is_odd, d=nums) == {'x':1,'z':3,'l':5}
        '''
        r = {}
        for key,v in d.items():
            if fn(v):
                r[key] = v
        return r
    keep_dict = filter_in_dict

    def filter_out(fn,l):
        '''
        ## FPO.filter_out(...)
        The inverse of FPO.filterIn(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new list. Otherwise, the value is included.
        ### Arguments:
            fn:     predicate function; called with v (value), i (index), and l (list) named arguments
            l:    list to filter against
        ### Returns:
            list
        ### Aliases:
            FPO.reject(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = [1,2,3,4,5]
            assert FPO.filter_out(fn=is_odd, l=nums) == [2,4]
        '''
        r = []
        for e in l:
            if fn(e) is not True:
               r.append(e)
        return r
    reject = filter_out

    def filter_out_dict(fn, d):
        '''
        ## FPO.filter_out_dict(...)
        The inverse of FPO.filterInObj(..), produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new object. Otherwise, the value is included at the same property name.
        ### Arguments:
            fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
            d:      dictionary to filter against
        ### Returns:
            dictionary
        ### Aliases:
            FPO.reject_dict(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
            assert FPO.filter_out_dict(fn=is_odd, d=nums) == {'y':2,'r':4}
        '''
        r = {}
        for key,v in d.items():
            if fn(v) != True:
                r[key] = v
        return r
    keep_dict = filter_out_dict

    def flat_map(fn,l):
        '''
        ## FPO.flat_map(...)
        Similar to map(..), produces a new list by calling a mapper function with each value in the original list. If the mapper function returns a list, this list is flattened (one level) into the overall list.
        ### Arguments:
            fn:  mapper function; called with v (value), i (index), and list(l) named arguments
            l:   list to flat-map against
        ### Returns:
            list
        ### Aliases:
            FPO.chain(..)
        ### Example:
            def split_chars(v): return [*v]
            words = ['hello','world']
            assert split_chars(v=words[0]) == ['h','e','l','l','o']
            assert list(map(split_chars, words)) == [['h','e','l','l','o'],['w','o','r','l','d']]
            assert FPO.flat_map(fn=split_chars, l=words) == ['h','e','l','l','o','w','o','r','l','d']
        '''
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

    def flat_map_dict(fn,d):
        '''
        ## FPO.flat_map_dict(...)
        ### Arguments:
            fn:     mapper function; called with v (value), i (property name), and d (dictionary) named arguments
            d:      dictionary to flat-map against
        ### Returns:
            dictionary
        ### Aliases:
            FPO.chain_dict(..)
        ### Example:
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
        dd = map_dict(fn,d)
        r = {}
        for key,v in dd.items():
            if isinstance(v, dict) is True:
                r.update(v)
            else:
                r[key] = v
        return r
    chain_dict = flat_map_dict

    def flatten(l, n=-1):
        '''
        ## FPO.flatten(...)
        Flattens an array of nested arrays. Optionally, specify how many levels of nesting to flatten out.
        ### Arguments:
            l:   list to flat-map against
            n:   (optional) the number of levels of nesting to flatten out; if omitted, defaults to Infinity (to flatten any nested depth)
        ### Returns:
            list
        ### Example:
            nums = [1,2,[3,4],[5,[6,7]]]
            assert FPO.flatten(l=nums) == [1,2,3,4,5,6,7]
            assert FPO.flatten(l=nums,n=1) == [1, 2, 3, 4, 5, [6, 7]]
        '''
        if n is 0: return l
        r = []
        for e in l:
            if isinstance(e, list) is True:
                r += flatten(e, n=(n-1))
            else:
                r.append(e)
        return r

    def head(v):
        '''
        ## FPO.head(...)
        Returns the element as accessed at index 0 of the value.
        ### Arguments:
            v:   list, tuple, dict, str
        ### Returns:
            any
        ### Example:
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
        if bool(v) is not True:
            return None
        elif isinstance(v, dict) is True:
            return next(iter(v.values()))
        elif isinstance(v, (list, tuple)) is True:
            return v[0]
        elif isinstance(v, str) is True:
            return list(v)[0]

    def identity(d):
        '''
        ## FPO.identity(...)
        Returns the value given to it. Useful as a default placeholder for certain operations(i.e., composition, reduction).
        ### Arguments:
            d:   list
        ### Returns:
            any
        ### Example:
            FPO.identity( {'v': 42} ) == 42
        See also: FPO.constant(...)
        '''
        return next(iter(d.values()))

    def map_dict(fn, d):
        '''
        ## FPO.map_dict(...)
        Produces a new dictionary by calling a mapper function with each property value in the original dictionary. The value the mapper function returns is inserted in the new object at that same property name. The new dictionary will always have the same number of properties as the original dictionary.
        ### Arguments:
            fn:     mapper function; called with v (value), i (index), and d (dictionary) named arguments
            d:   dictionary to-map against
        ### Returns:
            dictionary
        ### Example:
            def double(v, key): return v * 2
            nums = {'a': 1, 'b': 2, 'c': 3}
            assert FPO.map_dict(fn=double,d=nums) == {'a': 2, 'b': 4, 'c': 6}
        '''
        r = {}
        for key, v in d.items():
            r[key] = fn(v=v,key=key)
        return r

    def map_list(fn, l):
        '''
        ## FPO.map_list(...)
        Produces a new list by calling a mapper function with each value in the original list. The value the mapper function returns is inserted in the new list at that same position. The new list will always be the same length as the original list.
        ### Arguments:
            fn: mapper function; called with v (value) and l (list) named arguments
            l:  list to map against
        ### Returns:
            list
        ### Example:
            def double(v): return v * 2
            nums = [1,2,3]
            assert FPO.map_list(fn=double,l=nums) == [2,4,6]
        '''
        r = []
        for v in l:
            r.append(fn(v=v))
        return r

    def memoise(fn,n=-1):
        '''
        ## FPO.memoize(...)
        For performance optimization reasons, wraps a function such that it remembers each set of arguments passed to it, associated with that underlying return value. If the wrapped function is called subsequent times with the same set of arguments, the cached return value is returned instead of being recomputed. Each wrapped function instance has its own separate cache, even if wrapping the same original function multiple times.

        A set of arguments is "remembered" by being hashed to a string value to use as a cache key. This hashing is done internally with json.dumps(..), which is fast and works with many common value types. However, this hashing is by no means bullet-proof for all types, and does not guarantee collision-free. Use caution: generally, you should only use primitives (number, string, boolean, null, and None) or simple objects (dict, list) as arguments. If you use objects, always make sure to list properties in the same order to ensure proper hashing.

        Unary functions (single argument; n of 1) with a primitive argument are the fastest for memoisation, so if possible, try to design functions that way. In these cases, specifying n as 1 will help ensure the best possible performance.

        Warning: Be aware that if 1 is initially specified (or detected) for n, additional arguments later passed to the wrapped function are not considered in the memoisation hashing, though they will still be passed to the underlying function as-is. This may cause unexpected results (false-positives on cache hits); always make sure n matches the expected number of arguments.
        ### Arguments:
            fn: function to wrap
            n:  number of arguments to memoize; if omitted, tries to detect the arity (fn.length) to use.
        ### Returns:
            list
        ### Example:
            def sum(x,y):
                return x + y + random.randint(1,101)
            fa = FPO.memoise(fn=sum)
            fb = FPO.memoise(fn=sum, n=1)
            cached_a = fa(2,3)
            assert fa(2,3) == cached_a
            cached_b = fb(2,3)
            assert fb(2,4) == cached_b
        '''
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

    def n_ary(fn,props):
        '''
        ## FPO.n_ary(...)
        Wraps a function to restrict its inputs to only the named arguments as specified. It is similar to FPO.pluck.
        ### Arguments:
            fn:     function to wrap
            props:  list of property names to allow as named arguments; if empty, produces a "nullary" function -- won't receive any arguments.
        ### Returns:
            function
        ### Example:
            def foo(d): return d
            f = FPO.n_ary(fn=foo, props=['x','y','z'])
            assert f({'x': 1, 'y': 2, 'z': 3, 'w': 4}) == {'x': 1, 'y': 2, 'z': 3}
        '''
        def n_aried(d):
            if bool(props) is not True:
                return fn()
            else:
                r = {}
                for key in props:
                    r[key] = d[key]
                return fn(r)
        return n_aried

    def partial(fn, args):
        '''
        ## FPO.partial(...)
        Wraps a function with a new function that already has some of the arguments pre-specified, and is waiting for the rest of them on the next call. Unlike FPO.curry(..), you must specify all the remaining arguments on the next call of the partially-applied function.

        With traditional FP libraries, partial(..) works in left-to-right order (as does FPO.std.partial(..)). That's why typically you also need a FPO.std.partialRight(..) if you want to partially-apply from the opposite direction.

        However, using named arguments style -- after all, that is the whole point of FPO! -- order doesn't matter. For familiarity sake, FPO.partialRight(..) is provided, but it's just an alias to FPO.partial(..).
        ### Arguments:
            fn:     function to partially-apply
            args:   object containing the arguments to apply now
        ### Returns:
            function
        ### Example:
            def foo(x,y,z): return x + y + z
            f = FPO.partial(fn=foo, args={'x': 'a'});
            assert f(y='b', z='c') == 'abc'
        '''
        def partialed(**kwargs):
            l_kwargs = copy.copy(kwargs)
            l_kwargs.update(args)
            return fn(**l_kwargs)
        return partialed

    def pick(d,props):
        '''
        ## FPO.pick(...)
        Returns a new dictionary with only the specified properties from the original dictionary. Includes only properties from the original dictionary.
        ### Arguments:
            d:      dictionary to pick properties from
            props:  list of property names to pick from the object; if a property does not exist on the original dictionary, it is not added to the new dictionary, unlike FPO.pickAll(..).
        ### Returns:
            dictionary
        ### Example:
            d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
            assert FPO.pick(d,props=['x','y']) == {'x': 1, 'y': 2}
        '''
        r = {}
        for i in props:
            if i in d:
                r[i] = d[i]
        return r

    def pick_all(d, props):
        '''
        ## FPO.pick_all(...)
        Returns a new dictionary with only the specified properties from the original dictionary. Includes all specified properties.
        ### Arguments:
            d:      dictionary to pick properties from
            props:  list of property names to pick from the dictionary; even if a property does not exist on the original dictionary, it is still added to the new object with an undefined value, unlike FPO.pick(..).
        ### Returns:
            dictionary
        ### Example:
            d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
            assert FPO.pick_all(d,props=['x','y','r']) == {'x': 1, 'y': 2, 'r': None}
        '''
        r = {}
        for i in props:
            if i in d:
                r[i] = d[i]
            else:
                r[i] = None
        return r

    def pipe(fns):
        '''
        ## FPO.pipe(...)
        Produces a new function that's the composition of a list of functions. Functions are composed left-to-right (unlike FPO.compose(..)) from the array.
        ### Arguments:
            fns:    list of functions
        ### Returns:
            function
        ### Example:
            f = FPO.pipe([
                lambda v: v+2,
                lambda v: v*2,
                lambda v: v-2,
            ])
            assert f(10) == 22
        '''
        def piped(v):
            result = v
            for fn in fns:
                result = fn(v=result)
            return result
        return piped

    def pluck(l, *args):
        '''
        ## FPO.pluck(...)
        Plucks properties form the given list and return a list of properties' values
        ### Arguments:
            l:   list
            *args:  properties
        ### Returns:
            a list of values
        ### Example:
            l = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
            assert FPO.pluck(l, 'x', 'y') == [[1, 2], [3, 4]]
            assert FPO.pluck(l, 'x') == [1, 3]
        '''
        fn = lambda d, *args: [d[arg] for arg in args]
        r = [fn(o, *args) for o in l]
        if len(args) == 1:
            return [v[0] for v in r]
        else:
            return r

    def prop(d,prop):
        '''
        ## FPO.prop(...)
        Extracts a property's value from a dictionary.
        ### Arguments:
            d:    dictionary to pull the property value from
            prop: property name to pull from the dictionary
        ### Returns:
            any
        ### Example:
            obj = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
            assert FPO.prop(d=obj, prop='y') == 2
        '''
        return d[prop]

    def reassoc(d,props):
        '''
        ## FPO.reassoc(...)
        Like a mixture between FPO.pick(..) and FPO.setProp(..), creates a new dictionary that has properties remapped from original names to new names. Any properties present on the original dictionary that aren't remapped are copied with the same name.
        ### Arguments:
            d:      dictionary to remap properties from
            props:  dictionary whose key/value pairs are sourceProp: targetProp remappings
        ### Returns:
            dictionary
        ### Example:
            obj = dict(zip(['x','y','z'],[1, 2, 3]))
            assert FPO.reassoc(d=obj, props={'x': 'a', 'y': 'b'}) == {'a': 1, 'b': 2, 'z': 3}
            assert obj == {'x': 1, 'y': 2, 'z': 3}
        '''
        r = {}
        for k,v in d.items():
            if k in props:
                r[props[k]] = d[k]
            else:
                r[k] = d[k]
        return r

    def reduce(fn,l=[],v=None):
        '''
        ## FPO.reduce(..)
        Processes a list from left-to-right (unlike FPO.reduceRight(..)), successively combining (aka "reducing", "folding") two values into one, until the entire list has been reduced to a single value. An initial value for the reduction can optionally be provided.
        ### Arguments:
            fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
            l:  list to reduce
            v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
        ### Returns:
            any
        ### Example:
            def str_concat(acc,v):
                return acc + v
            vowels = ["a","e","i","o","u","y"]
            assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
            assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
            assert vowels == ["a","e","i","o","u","y"]
        '''
        orig_l = l
        initial_v = v
        if initial_v is None and len(l) > 0:
            initial_v = l[0]
            l = l[1:]
        for e in l:
            initial_v = fn(acc=initial_v, v=e)
        return initial_v

    def reduce_dict(fn,d,v=None):
        '''
        ## FPO.reduce_dict(..)
        Processes an dictionary's properties (in enumeration order), successively combining (aka "reducing", "folding") two values into one, until all the dictionary's properties have been reduced to a single value. An initial value for the reduction can optionally be provided.
        ### Arguments:
            fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
            d:  dictionary to reduce
            v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
        ### Returns:
            any
        ### Example:
            def str_concat(acc,v):
                return acc + v
            vowels = ["a","e","i","o","u","y"]
            assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
            assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
            assert vowels == ["a","e","i","o","u","y"]
        '''
        init_k = next(iter(d))
        r = d[init_k]
        for key,value in d.items():
            if key is not init_k:
                r = fn(acc=r, v=value)
        if bool(v) is True:
            return v + r
        return r

    def reduce_right(fn,l,v=None):
        '''
        ## FPO.reduce_right(..)
        Processes a list from right-to-left (unlike FPO.reduce(..)), successively combining (aka "reducing", "folding") two values into one, until the entire list has been reduced to a single value.
        An initial value for the reduction can optionally be provided. If the array is empty, the initial value is returned (or undefined if it was omitted).
        ### Arguments:
            fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
            l:  list to reduce
            v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
        ### Returns:
            any
        ### Example:
            def str_concat(acc,v):
                return acc + v
            vowels = ["a","e","i","o","u","y"]
            assert FPO.reduce_right(fn=str_concat, l=vowels) == 'yuoiea'
            assert FPO.reduce_right(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: yuoiea'
            assert vowels == ["a","e","i","o","u","y"]
        '''
        rl = l[::-1]
        r = rl[0]
        for e in rl[1:]:
            r = fn(acc=r, v=e)
        if bool(v) is True:
            return v + r
        return r

    def remap(fn, args):
        '''
        ## FPO.remap(..)
        Remaps the expected named arguments of a function. This is useful to adapt a function to be used if the arguments passed in will be different than what the function expects.
        A common usecase will be to adapt a function so it's suitable for use as a mapper/predicate/reducer function, or for composition.
        ### Arguments:
            fn:     function to remap
            args:   dictionary whose key/value pairs represent the origArgName: newArgName mappings
        ### Returns:
            function
        ### Example:
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
        def remaped(**kwargs):
            l_kwargs = reassoc(kwargs,props=args)
            return fn(**l_kwargs)
        return remaped

    def set_prop(d,prop,v):
        '''
        ##FPO.set_prop(...)
        Creates a shallow clone of a dictionary, assigning the specified property value to the new dictionary.
        ### Arguments:
            d:      (optional) object to clone; if omitted, defaults to a new empty dictionary
            prop:   property name where to set the value on the new dictionary
            v:      value
        ### Returns:
            any
        ### Example:
            obj = dict(x=1, y=2,z=3)
            assert FPO.set_prop(d=obj, prop='w', v=4) == {'x': 1, 'y': 2, 'z': 3, 'w': 4}
            assert obj == {'x': 1, 'y': 2, 'z': 3}
        '''
        if bool(d) is True:
            r = copy.copy(d)
        else:
            r = {}
        r[prop] = v
        return r

    def tail(v):
        '''
        ## FPO.tail(...)
        Returns everything else in the value except the element as accessed at index 0; basically the inverse of FPO.head(..)
        ### Arguments:
            v:   list/string/dictionary
        ### Returns:
            any
        ### Example:
            assert FPO.tail(v={'a':42,'b':56,'c':34}) == {'b':56,'c':34}
            assert FPO.tail(v=[1,2,3,4]) == [2,3,4]
            assert FPO.tail(v=(42,56,32)) == (56,32)
            assert FPO.tail(v='abc') == 'bc'
            assert FPO.tail(v=[]) == None
            assert FPO.tail(v={}) == None
            assert FPO.tail(v='') == None
        '''
        if bool(v) is not True:
            return None
        elif isinstance(v, dict) is True:
            init_k = next(iter(v))
            r = {}
            for key,value in v.items():
                if key is not init_k:
                    r[key] = value
            return r
        elif isinstance(v, (list, tuple)) is True:
            return v[1:]
        elif isinstance(v, str) is True:
            return v[1:]

    def take(iterable, n=1):
        '''
        ## FPO.take(...)
        Returns the specified number of elements from the value, starting from the beginning.
        ### Arguments:
            iterable:   list/string
            n:          number of elements to take from the beginning of the value; if omitted, defaults to `1`
        ### Returns:
            list/string
        ### Example:
            items = [2,4,6,8,10]
            assert FPO.take(items, 3) == [2,4,6]
            assert FPO.take(items) == [2]
            assert FPO.take({'apple','banana','cherry'}, 2) == ['apple','banana']
        '''
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

    def trampoline(fn):
        '''
        ## FPO.trampoline(...)
        Wraps a continuation-returning recursive function in another function that will run it until it no longer returns another continuation function. Trampolines are an alternative to tail calls.
        ### Arguments:
            fn:     function to run
        ### Returns:
            function
        ### Example:
            def sum(total,x):
                if x <= 1:
                return total + x
            return lambda : sum(total+x, x-1)
            assert FPO.trampoline(fn=sum)(0,5) == 15
        '''
        def trampolined(*args, **kwargs):
            if bool(args):
                r = fn(*args)
            else:
                r = fn(**kwargs)
            while callable(r) is True:
                r = r()
            return r
        return trampolined

    def transduce_fn(fn,co,v,l=[]):
        '''
        ## FPO.transducer_transduce(...)
        Produces a reducer from a specified transducer and combination function. Then runs a reduction on a list, using that reducer, starting with the specified initial value.
        Note: When composing transducers, the effective order of operations is reversed from normal composition. Instead of expecting composition to be right-to-left, the effective order will be left-to-right (see below).
        ### Arguments:
            fn: transducer function
            co: combination function for the transducer
            v: initial value for the reduction
            l: the list for the reduction
        ### Returns:
            any
        ### Example:
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
        '''
        transducer = fn
        combination_fn = co
        initial_value = v
        reducer = transducer(v=combination_fn)
        return reduce(fn=reducer, v=initial_value, l=l)
    transducer_transduce = curry_multiple(fn=transduce_fn, n=4)

    def transducer_map_fn(fn,v=None):
        '''
        ## FPO.transducer_map(...)
        For transducing purposes, wraps a mapper function as a map-transducer. Typically, this map-transducer is then composed with other filter-transducers and/or map-transducers. The resulting transducer is then passed to FPO.transducers.transduce(..).
        The map-transducer is not a reducer itself; it's expecting a combination function (reducer), which will then produce a filter-reducer. So alternately, you can manually create the map-reducer and use it directly with a regular FPO.reduce(..) reduction.
        ### Arguments:
            fn: mapper function
        ### Returns:
            function
        ### Example:
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
        '''
        mapper_fn = fn
        combination_fn = v
        #till waiting on the combination function?
        if combination_fn is None:
            #Note: the combination function is usually a composed
            #function, so we expect the argument by itself,
            #not wrapped in a dictionary
            def curried(v):
                nonlocal mapper_fn
                return transducer_map_fn(fn=mapper_fn,v=v)
            return curried

        def reducer(acc,v):
            nonlocal mapper_fn, combination_fn
            return combination_fn(acc,v=mapper_fn(v))
        return reducer
    transducer_map = curry_multiple(fn=transducer_map_fn, n=1)

    def transducer_filter_fn(fn,v=None):
        '''
        ## FPO.transducer_filter(...)
        For transducing purposes, wraps a predicate function as a filter-transducer. Typically, this filter-transducer is then composed with other filter-transducers and/or map-transducers. The resulting transducer is then passed to FPO.transducers.transduce(..).
        ### Arguments:
            fn:    predicate function
        ### Returns:
            function
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            def list_push(acc, v):
                acc.append(v)
                return acc
            nums = [1,2,3,4,5]
            filter_transducer = FPO.transducer_filter(fn=is_odd)
            r = FPO.transducer_transduce(fn=filter_transducer, co=list_push, v=[], l=nums)
            assert r == [1,3,5]
        '''
        predicated_fn = fn
        combination_fn = v
        #till waiting on the combination function?
        if combination_fn is None:
            #Note: the combination function is usually a composed
            #function, so we expect the argument by itself,
            #not wrapped in a dictionary
            def curried(v):
                nonlocal predicated_fn
                return transducer_filter_fn(fn=predicated_fn,v=v)
            return curried

        def reducer(acc,v):
            nonlocal predicated_fn, combination_fn
            if predicated_fn(v):
                return combination_fn(acc, v)
            return acc
        return reducer
    transducer_filter = curry_multiple(fn=transducer_filter_fn, n=1)

    def transducer_into_fn(fn,v,l):
        '''
        ## FPO.transducer_into(...)
        Selects an appropriate combination function (reducer) based on the provided initial value. Then runs FPO.transducers.transduce(..) under the covers.

        Detects initial values of boolean, number, string, and list types, and dispatches to the appropriate combination function accordingly (FPO.transducers.number(..), etc). Note: A boolean initial value selects FPO.transducer_bool_and(..).

        Note: When composing transducers, the effective order of operations is reversed from normal composition. Instead of expecting composition to be right-to-left, the effective order will be left-to-right (see below).
        ### Arguments:
            fn: transducer function
            v:  initial value for the reduction; also used to select the appropriate combination function (reducer) for the transducing.
            l: the list for the reductiontransduce_fn
        ### Example:
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
        '''
        transducer = fn
        combination_fn = transducer_default
        if isinstance(v, bool):
            combination_fn = transducer_bool_and
        elif isinstance(v, str):
            combination_fn = transducer_string
        elif isinstance(v, int):
            combination_fn = transducer_number
        elif isinstance(v, list):
            combination_fn = transducer_list
        else:
            transducer_default
        return transduce_fn(fn=transducer, co=combination_fn, v=v, l=l)
    transducer_into = curry_multiple(fn=transducer_into_fn, n=3)

    def transducer_default(acc,v):
        '''
        ## FPO.transducer_default(...)
        A reducer function. For transducing purposes, a combination function that's a default placeholder. It returns only the acc that's passed to it. The behavior here is almost the same as FPO.identity(..), except that returns acc instead of v.
        ### Arguments:
            acc:    acculumator
            v:  value
        ### Returns:
            any
        ### Example:
            assert FPO.transducer_default(acc=3, v=1) == 3
        '''
        return acc

    def transducer_list(acc,v):
        '''
        ## FPO.transducer_list(...)
        A reducer function. For transducing purposes, a combination function that takes an array and a value, and mutates the array by pushing the value onto the end of it. The mutated array is returned.
        *This function has side-effects*, for performance reasons. It should be used with caution.
        ### Arguments:
            acc:    acculumator
            v:  value
        ### Returns:
            list
        ### Example:
            arr = [1,2,3]
            FPO.transducer_list(acc=arr,v=4)
            assert arr == [1,2,3,4]
        '''
        acc.append(v)
        return acc

    def transducer_bool_and(acc,v):
        '''
        ## FPO.transducer_bool_and(...)
        A reducer function. For transducing purposes, a combination function that takes two booleans and ANDs them together. The result is the logical AND of the two values.
        ### Arguments:
            acc:    acculumator
            v:  value
        ### Returns:
            true/false
        ### Example:
            assert FPO.transducer_bool_and(acc=True, v=True) == True
            assert FPO.transducer_bool_and(acc=False, v=True) == False
        '''
        if bool(acc) and bool(v) is True:
            return True
        else:
            return False

    def transducer_bool_or(acc,v):
        '''
        ## FPO.transducer_bool_or(...)
        A reducer function. For transducing purposes, a combination function that takes two booleans and ORs them together. The result is the logical OR of the two values.
        ### Arguments:
            acc:    acculumator
            v:  value
        ### Returns:
            true/false
        ### Example:
            assert FPO.transducer_bool_or(acc=True, v=True) == True
            assert FPO.transducer_bool_or(acc=False, v=False) == False
            assert FPO.transducer_bool_or(acc=False, v=True) == True
        '''
        if bool(acc) or bool(v) is True:
            return True
        else:
            return False

    def transducer_number(acc,v):
        '''
        ## FPO.transducer_number(...)
        A reducer function. For transducing purposes, a combination function that adds together the two numbers passed into it. The result is the sum.
        ### Arguments:
            acc: acculumator
            v: value
        ### Returns:
            number
        ### Example:
            assert FPO.transducer_number( acc=3, v=4) == 7
        '''
        return acc + v

    def transducer_string(acc,v):
        '''
        ## FPO.transducer_string(...)
        A reducer function. For transducing purposes, a combination function that concats the two strings passed into it. The result is the concatenation.
        ### Arguments:
            acc: acculumator
            v: value
        ### Returns:
            string
        ### Example:
            assert FPO.transducer_string( acc='hello', v='world') == 'helloworld'
        '''
        return str(acc) + str(v)

    def unapply(fn, props):
        '''
        ## FPO.unapply(..)
        Wraps a function to gather individual positional arguments into an object argument.
        ### Arguments:
            fn:     function to wrap
            props:  list of property names (strings) to indicate the order to gather individual positional arguments as properties.
        ### Returns:
            function
        Example:
            def foo(x,y):
                return x + y
            f = FPO.unapply(fn=foo, props=['x','y'])
            assert f(1,2) == 3
        '''
        def unapplied(*args):
            g = zip(props,args)
            kwargs = dict(g)
            return fn(**kwargs)
        return unapplied

    def unary(fn,prop):
        '''
        ## FPO.unary(..)
        Wraps a function to restrict its inputs to only one named argument as specified.
        ### Arguments:
            fn: function to wrap
            prop: property name to allow as named argument
        ### Returns:
            function
        ### Example:
            def foo(**kwargs):
                return kwargs
            f = FPO.unary(fn=foo, prop='y')
            assert f(x=1,y=2,z=3) == {'y':2}
        '''
        def unary_fn(**kwargs):
            l_kwargs = {}
            l_kwargs[prop] = kwargs[prop]
            return fn(**l_kwargs)
        return unary_fn

    def uncurry(fn):
        '''
        ## FPO.uncurry(...)
        Wraps a (strictly) curried function in a new function that accepts all the arguments at once, and provides them one at a time to the underlying curried function.
        ### Arguments:
            fn: function to uncurry
        ### Returns:
            function
        ### Example:
            def foo(x,y,z):
                return x + y + z
            f = FPO.curry(fn=foo, n=3)
            p = FPO.uncurry(fn=f)
            assert p(x=1,y=2,z=3) == 6
        '''
        def uncurry_fn(**kwargs):
            print('AAAA', kwargs)
            r = fn
            for key,v in kwargs.items():
                r = r(**{key:v})
            return r
        return uncurry_fn

</details></section>

<section>

## Functions

<dl>

<dt id="fpo.ap">`<span>def <span class="ident">ap</span></span>(<span>fns, list)</span>`</dt>

<dd>

<section class="desc">

## FPO.ap(…)

Produces a new list that is a concatenation of sub-lists, each produced by calling FPO.map(..) with each mapper function and the main list.

### Arguments:

    fns:   a list of functions
    list:  a list

### Returns:

    a list of new values

### Example:

    fn1 = lambda increment(v) { return v + 1 }
    def double(v) { return v * 2 }
    nums = [1,2,3,4,5]

### Reference:

    # <https://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/>

</section>

<details class="source"><summary>Source code</summary>

    def ap(fns, list):
        '''
        ## FPO.ap(...)
        Produces a new list that is a concatenation of sub-lists, each produced by calling FPO.map(..) with each mapper function and the main list.
        ### Arguments:
            fns:   a list of functions
            list:  a list
        ### Returns:
            a list of new values
        ### Example:
            fn1 = lambda increment(v) { return v + 1 }
            def double(v) { return v * 2 }
            nums = [1,2,3,4,5]
        ### Reference:
            # https://treyhunner.com/2015/12/python-list-comprehensions-now-in-color/
        '''
        matrix = [[fn(v) for v in list] for fn in fns]
        return [n for row in matrix for n in row]

</details></dd>

<dt id="fpo.apply">`<span>def <span class="ident">apply</span></span>(<span>fn, props=None)</span>`</dt>

<dd>

<section class="desc">

## FPO.apply(…)

Wraps a function to spread out the properties from an object arugment as individual positional arguments

### Arguments:

    fn:     function to wrap
    props:  (optional) list of property names (strings) to indicate the order to spread properties as individual arguments. If omitted, the signature of fn is parsed for its parameter list to try to determine an ordered property list; this detection only works for simple parameters (including those with default parameter value settings).

### Returns:

    function

### Example:

    def foo(x, y=2): return x + y
    def bar(a, b, c=0): return a + b + c
    f = FPO.apply(fn=foo)
    p = FPO.apply(fn=bar, props=['x','y'])
    assert f({'a': 1, 'b':1}) == 2
    assert f({'x': 3}) == 5
    assert p({'x': 3, 'y': 2}) == 5

</section>

<details class="source"><summary>Source code</summary>

    def apply(fn, props=None):
        '''
        ## FPO.apply(...)
        Wraps a function to spread out the properties from an object arugment as individual positional arguments
        ### Arguments:
            fn:     function to wrap
            props:  (optional) list of property names (strings) to indicate the order to spread properties as individual arguments. If omitted, the signature of fn is parsed for its parameter list to try to determine an ordered property list; this detection only works for simple parameters (including those with default parameter value settings).
        ### Returns:
            function
        ### Example:
            def foo(x, y=2): return x + y
            def bar(a, b, c=0): return a + b + c
            f = FPO.apply(fn=foo)
            p = FPO.apply(fn=bar, props=['x','y'])
            assert f({'a': 1, 'b':1}) == 2
            assert f({'x': 3}) == 5
            assert p({'x': 3, 'y': 2}) == 5
        '''
        def applied(d):
            if props is None:
                return fn(*(d[key] for key,v in d.items()))
            return fn(*(d[key] for key in props))
        return applied

</details></dd>

<dt id="fpo.binary">`<span>def <span class="ident">binary</span></span>(<span>fn, props)</span>`</dt>

<dd>

<section class="desc">

## FPO.binary(…)

Wraps a function to restrict its inputs to dictionary with only two named arguments as specified.

### Arguments:

    fn:     function to wrap
    props:  list of two property names to allow as named arguments

### Returns:

    function

### Example:

    def foo(x,y): return x + y
    def bar(a,b,c=1): return a + b + c
    f = FPO.binary(fn=foo, props=['x','y'])
    p = FPO.binary(fn=bar, props=['a','b'])
    assert f({'x':1, 'y':2, 'z':4}) == 3
    assert p({'a':2,'b':4,'c':6}) == 7

</section>

<details class="source"><summary>Source code</summary>

    def binary(fn,props):
        '''
        ## FPO.binary(...)
        Wraps a function to restrict its inputs to dictionary with only two named arguments as specified.

        ### Arguments:
            fn:     function to wrap
            props:  list of two property names to allow as named arguments
        ### Returns:
            function
        ### Example:
            def foo(x,y): return x + y
            def bar(a,b,c=1): return a + b + c
            f = FPO.binary(fn=foo, props=['x','y'])
            p = FPO.binary(fn=bar, props=['a','b'])
            assert f({'x':1, 'y':2, 'z':4}) == 3
            assert p({'a':2,'b':4,'c':6}) == 7
        '''
        _props = props[slice(0,2)]
        ln = lambda d, props: {key:d[key] for key in props}
        def binaryFn(d):
            return fn(**ln(d, _props))
        return binaryFn

</details></dd>

<dt id="fpo.chain">`<span>def <span class="ident">chain</span></span>(<span>fn, l)</span>`</dt>

<dd>

<section class="desc">

## FPO.flat_map(…)

Similar to map(..), produces a new list by calling a mapper function with each value in the original list. If the mapper function returns a list, this list is flattened (one level) into the overall list.

### Arguments:

    fn:  mapper function; called with v (value), i (index), and list(l) named arguments
    l:   list to flat-map against

### Returns:

    list

### Aliases:

    FPO.chain(..)

### Example:

    def split_chars(v): return [*v]
    words = ['hello','world']
    assert split_chars(v=words[0]) == ['h','e','l','l','o']
    assert list(map(split_chars, words)) == [['h','e','l','l','o'],['w','o','r','l','d']]
    assert FPO.flat_map(fn=split_chars, l=words) == ['h','e','l','l','o','w','o','r','l','d']

</section>

<details class="source"><summary>Source code</summary>

    def flat_map(fn,l):
        '''
        ## FPO.flat_map(...)
        Similar to map(..), produces a new list by calling a mapper function with each value in the original list. If the mapper function returns a list, this list is flattened (one level) into the overall list.
        ### Arguments:
            fn:  mapper function; called with v (value), i (index), and list(l) named arguments
            l:   list to flat-map against
        ### Returns:
            list
        ### Aliases:
            FPO.chain(..)
        ### Example:
            def split_chars(v): return [*v]
            words = ['hello','world']
            assert split_chars(v=words[0]) == ['h','e','l','l','o']
            assert list(map(split_chars, words)) == [['h','e','l','l','o'],['w','o','r','l','d']]
            assert FPO.flat_map(fn=split_chars, l=words) == ['h','e','l','l','o','w','o','r','l','d']
        '''
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

</details></dd>

<dt id="fpo.chain_dict">`<span>def <span class="ident">chain_dict</span></span>(<span>fn, d)</span>`</dt>

<dd>

<section class="desc">

## FPO.flat_map_dict(…)

### Arguments:

    fn:     mapper function; called with v (value), i (property name), and d (dictionary) named arguments
    d:      dictionary to flat-map against

### Returns:

    dictionary

### Aliases:

    FPO.chain_dict(..)

### Example:

    def split_evens_in_half(v, key):
        if v % 2 == 0:
        return { key: v/2, key+'_2': v/2 }
    return v
    nums = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert split_evens_in_half(v=3, key='c') == 3
    assert split_evens_in_half(v=4, key='d') == {'d':2, 'd_2': 2 }
    assert FPO.map_dict(fn=split_evens_in_half, d=nums) == {'a': 1, 'b': {'b': 1, 'b_2': 1}, 'c': 3, 'd': {'d': 2, 'd_2': 2}}
    assert FPO.flat_map_dict(fn=split_evens_in_half, d=nums) == {'a': 1, 'b': 1, 'b_2': 1, 'c': 3, 'd': 2, 'd_2': 2}

</section>

<details class="source"><summary>Source code</summary>

    def flat_map_dict(fn,d):
        '''
        ## FPO.flat_map_dict(...)
        ### Arguments:
            fn:     mapper function; called with v (value), i (property name), and d (dictionary) named arguments
            d:      dictionary to flat-map against
        ### Returns:
            dictionary
        ### Aliases:
            FPO.chain_dict(..)
        ### Example:
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
        dd = map_dict(fn,d)
        r = {}
        for key,v in dd.items():
            if isinstance(v, dict) is True:
                r.update(v)
            else:
                r[key] = v
        return r

</details></dd>

<dt id="fpo.complement">`<span>def <span class="ident">complement</span></span>(<span>fn)</span>`</dt>

<dd>

<section class="desc">

## FPO.complement(…)

Wraps a predicate function – a function that produces true / false – to negate its result.

### Arguments:

    fn:     function to wrap

### Returns:

    function

### Example:

    def foo(x,y): return x > y
    def bar(): return True
    f = FPO.complement(foo)
    p = FPO.complement(bar)
    assert foo(3,2) == True
    assert f(3,2) == False
    assert bar() == True
    assert p() == False

</section>

<details class="source"><summary>Source code</summary>

    def complement(fn):
        '''
        ## FPO.complement(...)
        Wraps a predicate function -- a function that produces true / false -- to negate its result.
        ### Arguments:
            fn:     function to wrap
        ### Returns:
            function
        ### Example:
            def foo(x,y): return x > y
            def bar(): return True
            f = FPO.complement(foo)
            p = FPO.complement(bar)
            assert foo(3,2) == True
            assert f(3,2) == False
            assert bar() == True
            assert p() == False
        '''
        def complemented(*arg):
            return False if fn(*arg) is True else True
        return complemented

</details></dd>

<dt id="fpo.compose">`<span>def <span class="ident">compose</span></span>(<span>fns)</span>`</dt>

<dd>

<section class="desc">

## FPO.compose(…)

Produces a new function that's the composition of a list of functions. Functions are composed right-to-left (unlike FPO.pipe(..)) from the list.

### Arguments:

    fns:     list of (lambda)functions

### Returns:

    function

### Example:

    f = FPO.compose([
        lambda v: v+2,
        lambda v: v*2,
        lambda v: v-2,
    ])
    assert f(10) == 18

</section>

<details class="source"><summary>Source code</summary>

    def compose(fns):
        '''
        ## FPO.compose(...)
        Produces a new function that's the composition of a list of functions. Functions are composed right-to-left (unlike FPO.pipe(..)) from the list.
        ### Arguments:
            fns:     list of (lambda)functions
        ### Returns:
            function
        ### Example:
            f = FPO.compose([
                lambda v: v+2,
                lambda v: v*2,
                lambda v: v-2,
            ])
            assert f(10) == 18
        '''
        def composed(v):
            result = v
            for fn in reversed(fns):
                result = fn(v=result)
            return result
        return composed

</details></dd>

<dt id="fpo.constant">`<span>def <span class="ident">constant</span></span>(<span>v)</span>`</dt>

<dd>

<section class="desc">

## FPO.constant(…)

Wraps a value in a fureversed

### Arguments:

    v:     constant vreversed

### Returns:

    function

### Example:

    f = FPO.constant(12)
    assert f() == 12
    assert f(24,9) == 12
    assert f(24) == 12

</section>

<details class="source"><summary>Source code</summary>

    def constant(v):
        '''
        ## FPO.constant(...)
        Wraps a value in a fureversed
        ### Arguments:
            v:     constant vreversed
        ### Returns:
            function
        ### Example:
            f = FPO.constant(12)
            assert f() == 12
            assert f(24,9) == 12
            assert f(24) == 12
        '''
        def fn(*arg):
            return v
        return fn

</details></dd>

<dt id="fpo.curry">`<span>def <span class="ident">curry</span></span>(<span>fn, n)</span>`</dt>

<dd>

<section class="desc">

## FPO.curry(…)

Curries a function so that you can pass one argument at a time, each time getting back another function to receive the next argument. Once all arguments are passed, the underlying function is called with the arguments.

Unlike FPO.curryMultiple(..), you can only pass one property argument at a time to each curried function (see example below). If multiple properties are passed to a curried call, only the first property (in enumeration order) will be passed.

### Arguments:

    fn:     function to curry
    n:      number of arguments to curry for

### Returns:

    function

### Example:

    def foo(x,y,z):
        return x + y + z
    f = FPO.curry(fn=foo, n=3)
    v = f(x=1)()(y=2, z=3)(z=4)
    assert v == 7

</section>

<details class="source"><summary>Source code</summary>

    def curry(fn, n):
        '''
        ## FPO.curry(...)
        Curries a function so that you can pass one argument at a time, each time getting back another function to receive the next argument. Once all arguments are passed, the underlying function is called with the arguments.

        Unlike FPO.curryMultiple(..), you can only pass one property argument at a time to each curried function (see example below). If multiple properties are passed to a curried call, only the first property (in enumeration order) will be passed.
        ### Arguments:
            fn:     function to curry
            n:      number of arguments to curry for
        ### Returns:
            function
        ### Example:
            def foo(x,y,z):
                return x + y + z
            f = FPO.curry(fn=foo, n=3)
            v = f(x=1)()(y=2, z=3)(z=4)
            assert v == 7
        '''
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

</details></dd>

<dt id="fpo.curry_multiple">`<span>def <span class="ident">curry_multiple</span></span>(<span>fn, n)</span>`</dt>

<dd>

<section class="desc">

## FPO.curry_multiple(…)

Just like FPO.curry(..), except each curried function allows multiple arguments instead of just one.

Unlike FPO.curryMultiple(..), you can only pass one property argument at a time to each curried function (see example below). If multiple properties are passed to a curried call, only the first property (in enumeration order) will be passed.

### Arguments:

    fn:     function to curry
    n:      number of arguments to curry for

### Returns:

    function

### Example:

    def foo(x,y,z):
        return x + y + z
    f = FPO.curry_multiple(fn=foo, n=3)
    v = f(x=0,y=1)()(x=1)(y=2,z=3)
    assert v == 6

</section>

<details class="source"><summary>Source code</summary>

    def curry_multiple(fn, n):
        '''
        ## FPO.curry_multiple(...)
        Just like FPO.curry(..), except each curried function allows multiple arguments instead of just one.

        Unlike FPO.curryMultiple(..), you can only pass one property argument at a time to each curried function (see example below). If multiple properties are passed to a curried call, only the first property (in enumeration order) will be passed.
        ### Arguments:
            fn:     function to curry
            n:      number of arguments to curry for
        ### Returns:
            function
        ### Example:
            def foo(x,y,z):
                return x + y + z
            f = FPO.curry_multiple(fn=foo, n=3)
            v = f(x=0,y=1)()(x=1)(y=2,z=3)
            assert v == 6
        '''
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

</details></dd>

<dt id="fpo.filter_in">`<span>def <span class="ident">filter_in</span></span>(<span>fn, l)</span>`</dt>

<dd>

<section class="desc">

## FPO.filter_in(…)

Commonly known as filter(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new list. Otherwise, the value is omitted. It is the same as python filter() method

### Arguments:

    fn:     predicate function; called with v (value), i (index), and l (list) named arguments
    l:    list to filter against

### Returns:

    list

### Aliases:

    FPO.keep(..)

### Example:

    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_in(fn=is_odd, l=nums) == [1,3,5]

</section>

<details class="source"><summary>Source code</summary>

    def filter_in(fn,l):
        '''
        ## FPO.filter_in(...)
        Commonly known as filter(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new list. Otherwise, the value is omitted.
        It is the same as python filter() method
        ### Arguments:
            fn:     predicate function; called with v (value), i (index), and l (list) named arguments
            l:    list to filter against
        ### Returns:
            list
        ### Aliases:
            FPO.keep(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = [1,2,3,4,5]
            assert FPO.filter_in(fn=is_odd, l=nums) == [1,3,5]
        '''
        r = []
        for e in l:
            if fn(e):
               r.append(e)
        return r

</details></dd>

<dt id="fpo.filter_in_dict">`<span>def <span class="ident">filter_in_dict</span></span>(<span>fn, d)</span>`</dt>

<dd>

<section class="desc">

## FPO.filter_in_dict(…)

Produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new object at the same property name. Otherwise, the value is omitted.

### Arguments:

    fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
    d:      dictionary to filter against

### Returns:

    dictionary

### Aliases:

    FPO.keep_dict(..)

### Example:

    def is_odd(v):
        return v % 2 == 1
    nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
    assert FPO.filter_in_dict(fn=is_odd, d=nums) == {'x':1,'z':3,'l':5}

</section>

<details class="source"><summary>Source code</summary>

    def filter_in_dict(fn, d):
        '''
        ## FPO.filter_in_dict(...)
        Produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new object at the same property name. Otherwise, the value is omitted.
        ### Arguments:
            fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
            d:      dictionary to filter against
        ### Returns:
            dictionary
        ### Aliases:
            FPO.keep_dict(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
            assert FPO.filter_in_dict(fn=is_odd, d=nums) == {'x':1,'z':3,'l':5}
        '''
        r = {}
        for key,v in d.items():
            if fn(v):
                r[key] = v
        return r

</details></dd>

<dt id="fpo.filter_out">`<span>def <span class="ident">filter_out</span></span>(<span>fn, l)</span>`</dt>

<dd>

<section class="desc">

## FPO.filter_out(…)

The inverse of FPO.filterIn(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new list. Otherwise, the value is included.

### Arguments:

    fn:     predicate function; called with v (value), i (index), and l (list) named arguments
    l:    list to filter against

### Returns:

    list

### Aliases:

    FPO.reject(..)

### Example:

    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_out(fn=is_odd, l=nums) == [2,4]

</section>

<details class="source"><summary>Source code</summary>

    def filter_out(fn,l):
        '''
        ## FPO.filter_out(...)
        The inverse of FPO.filterIn(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new list. Otherwise, the value is included.
        ### Arguments:
            fn:     predicate function; called with v (value), i (index), and l (list) named arguments
            l:    list to filter against
        ### Returns:
            list
        ### Aliases:
            FPO.reject(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = [1,2,3,4,5]
            assert FPO.filter_out(fn=is_odd, l=nums) == [2,4]
        '''
        r = []
        for e in l:
            if fn(e) is not True:
               r.append(e)
        return r

</details></dd>

<dt id="fpo.filter_out_dict">`<span>def <span class="ident">filter_out_dict</span></span>(<span>fn, d)</span>`</dt>

<dd>

<section class="desc">

## FPO.filter_out_dict(…)

The inverse of FPO.filterInObj(..), produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new object. Otherwise, the value is included at the same property name.

### Arguments:

    fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
    d:      dictionary to filter against

### Returns:

    dictionary

### Aliases:

    FPO.reject_dict(..)

### Example:

    def is_odd(v):
        return v % 2 == 1
    nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
    assert FPO.filter_out_dict(fn=is_odd, d=nums) == {'y':2,'r':4}

</section>

<details class="source"><summary>Source code</summary>

    def filter_out_dict(fn, d):
        '''
        ## FPO.filter_out_dict(...)
        The inverse of FPO.filterInObj(..), produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new object. Otherwise, the value is included at the same property name.
        ### Arguments:
            fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
            d:      dictionary to filter against
        ### Returns:
            dictionary
        ### Aliases:
            FPO.reject_dict(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
            assert FPO.filter_out_dict(fn=is_odd, d=nums) == {'y':2,'r':4}
        '''
        r = {}
        for key,v in d.items():
            if fn(v) != True:
                r[key] = v
        return r

</details></dd>

<dt id="fpo.flat_map">`<span>def <span class="ident">flat_map</span></span>(<span>fn, l)</span>`</dt>

<dd>

<section class="desc">

## FPO.flat_map(…)

Similar to map(..), produces a new list by calling a mapper function with each value in the original list. If the mapper function returns a list, this list is flattened (one level) into the overall list.

### Arguments:

    fn:  mapper function; called with v (value), i (index), and list(l) named arguments
    l:   list to flat-map against

### Returns:

    list

### Aliases:

    FPO.chain(..)

### Example:

    def split_chars(v): return [*v]
    words = ['hello','world']
    assert split_chars(v=words[0]) == ['h','e','l','l','o']
    assert list(map(split_chars, words)) == [['h','e','l','l','o'],['w','o','r','l','d']]
    assert FPO.flat_map(fn=split_chars, l=words) == ['h','e','l','l','o','w','o','r','l','d']

</section>

<details class="source"><summary>Source code</summary>

    def flat_map(fn,l):
        '''
        ## FPO.flat_map(...)
        Similar to map(..), produces a new list by calling a mapper function with each value in the original list. If the mapper function returns a list, this list is flattened (one level) into the overall list.
        ### Arguments:
            fn:  mapper function; called with v (value), i (index), and list(l) named arguments
            l:   list to flat-map against
        ### Returns:
            list
        ### Aliases:
            FPO.chain(..)
        ### Example:
            def split_chars(v): return [*v]
            words = ['hello','world']
            assert split_chars(v=words[0]) == ['h','e','l','l','o']
            assert list(map(split_chars, words)) == [['h','e','l','l','o'],['w','o','r','l','d']]
            assert FPO.flat_map(fn=split_chars, l=words) == ['h','e','l','l','o','w','o','r','l','d']
        '''
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

</details></dd>

<dt id="fpo.flat_map_dict">`<span>def <span class="ident">flat_map_dict</span></span>(<span>fn, d)</span>`</dt>

<dd>

<section class="desc">

## FPO.flat_map_dict(…)

### Arguments:

    fn:     mapper function; called with v (value), i (property name), and d (dictionary) named arguments
    d:      dictionary to flat-map against

### Returns:

    dictionary

### Aliases:

    FPO.chain_dict(..)

### Example:

    def split_evens_in_half(v, key):
        if v % 2 == 0:
        return { key: v/2, key+'_2': v/2 }
    return v
    nums = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert split_evens_in_half(v=3, key='c') == 3
    assert split_evens_in_half(v=4, key='d') == {'d':2, 'd_2': 2 }
    assert FPO.map_dict(fn=split_evens_in_half, d=nums) == {'a': 1, 'b': {'b': 1, 'b_2': 1}, 'c': 3, 'd': {'d': 2, 'd_2': 2}}
    assert FPO.flat_map_dict(fn=split_evens_in_half, d=nums) == {'a': 1, 'b': 1, 'b_2': 1, 'c': 3, 'd': 2, 'd_2': 2}

</section>

<details class="source"><summary>Source code</summary>

    def flat_map_dict(fn,d):
        '''
        ## FPO.flat_map_dict(...)
        ### Arguments:
            fn:     mapper function; called with v (value), i (property name), and d (dictionary) named arguments
            d:      dictionary to flat-map against
        ### Returns:
            dictionary
        ### Aliases:
            FPO.chain_dict(..)
        ### Example:
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
        dd = map_dict(fn,d)
        r = {}
        for key,v in dd.items():
            if isinstance(v, dict) is True:
                r.update(v)
            else:
                r[key] = v
        return r

</details></dd>

<dt id="fpo.flatten">`<span>def <span class="ident">flatten</span></span>(<span>l, n=-1)</span>`</dt>

<dd>

<section class="desc">

## FPO.flatten(…)

Flattens an array of nested arrays. Optionally, specify how many levels of nesting to flatten out.

### Arguments:

    l:   list to flat-map against
    n:   (optional) the number of levels of nesting to flatten out; if omitted, defaults to Infinity (to flatten any nested depth)

### Returns:

    list

### Example:

    nums = [1,2,[3,4],[5,[6,7]]]
    assert FPO.flatten(l=nums) == [1,2,3,4,5,6,7]
    assert FPO.flatten(l=nums,n=1) == [1, 2, 3, 4, 5, [6, 7]]

</section>

<details class="source"><summary>Source code</summary>

    def flatten(l, n=-1):
        '''
        ## FPO.flatten(...)
        Flattens an array of nested arrays. Optionally, specify how many levels of nesting to flatten out.
        ### Arguments:
            l:   list to flat-map against
            n:   (optional) the number of levels of nesting to flatten out; if omitted, defaults to Infinity (to flatten any nested depth)
        ### Returns:
            list
        ### Example:
            nums = [1,2,[3,4],[5,[6,7]]]
            assert FPO.flatten(l=nums) == [1,2,3,4,5,6,7]
            assert FPO.flatten(l=nums,n=1) == [1, 2, 3, 4, 5, [6, 7]]
        '''
        if n is 0: return l
        r = []
        for e in l:
            if isinstance(e, list) is True:
                r += flatten(e, n=(n-1))
            else:
                r.append(e)
        return r

</details></dd>

<dt id="fpo.head">`<span>def <span class="ident">head</span></span>(<span>v)</span>`</dt>

<dd>

<section class="desc">

## FPO.head(…)

Returns the element as accessed at index 0 of the value.

### Arguments:

    v:   list, tuple, dict, str

### Returns:

    any

### Example:

    nums = [1,2,3,4]
    assert FPO.head(v={'a':42,'b':56}) == 42
    assert FPO.head(v=nums) == 1
    assert FPO.head(v=(42,56)) == 42
    assert FPO.head(v='abc') == 'a'
    assert FPO.head(v=[]) == None
    assert FPO.head(v={}) == None
    assert FPO.head(v='') == None

</section>

<details class="source"><summary>Source code</summary>

    def head(v):
        '''
        ## FPO.head(...)
        Returns the element as accessed at index 0 of the value.
        ### Arguments:
            v:   list, tuple, dict, str
        ### Returns:
            any
        ### Example:
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
        if bool(v) is not True:
            return None
        elif isinstance(v, dict) is True:
            return next(iter(v.values()))
        elif isinstance(v, (list, tuple)) is True:
            return v[0]
        elif isinstance(v, str) is True:
            return list(v)[0]

</details></dd>

<dt id="fpo.identity">`<span>def <span class="ident">identity</span></span>(<span>d)</span>`</dt>

<dd>

<section class="desc">

## FPO.identity(…)

Returns the value given to it. Useful as a default placeholder for certain operations(i.e., composition, reduction).

### Arguments:

    d:   list

### Returns:

    any

### Example:

    FPO.identity( {'v': 42} ) == 42

See also: FPO.constant(…)

</section>

<details class="source"><summary>Source code</summary>

    def identity(d):
        '''
        ## FPO.identity(...)
        Returns the value given to it. Useful as a default placeholder for certain operations(i.e., composition, reduction).
        ### Arguments:
            d:   list
        ### Returns:
            any
        ### Example:
            FPO.identity( {'v': 42} ) == 42
        See also: FPO.constant(...)
        '''
        return next(iter(d.values()))

</details></dd>

<dt id="fpo.keep">`<span>def <span class="ident">keep</span></span>(<span>fn, l)</span>`</dt>

<dd>

<section class="desc">

## FPO.filter_in(…)

Commonly known as filter(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new list. Otherwise, the value is omitted. It is the same as python filter() method

### Arguments:

    fn:     predicate function; called with v (value), i (index), and l (list) named arguments
    l:    list to filter against

### Returns:

    list

### Aliases:

    FPO.keep(..)

### Example:

    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_in(fn=is_odd, l=nums) == [1,3,5]

</section>

<details class="source"><summary>Source code</summary>

    def filter_in(fn,l):
        '''
        ## FPO.filter_in(...)
        Commonly known as filter(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is included in (aka, filtered into) the new list. Otherwise, the value is omitted.
        It is the same as python filter() method
        ### Arguments:
            fn:     predicate function; called with v (value), i (index), and l (list) named arguments
            l:    list to filter against
        ### Returns:
            list
        ### Aliases:
            FPO.keep(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = [1,2,3,4,5]
            assert FPO.filter_in(fn=is_odd, l=nums) == [1,3,5]
        '''
        r = []
        for e in l:
            if fn(e):
               r.append(e)
        return r

</details></dd>

<dt id="fpo.keep_dict">`<span>def <span class="ident">keep_dict</span></span>(<span>fn, d)</span>`</dt>

<dd>

<section class="desc">

## FPO.filter_out_dict(…)

The inverse of FPO.filterInObj(..), produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new object. Otherwise, the value is included at the same property name.

### Arguments:

    fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
    d:      dictionary to filter against

### Returns:

    dictionary

### Aliases:

    FPO.reject_dict(..)

### Example:

    def is_odd(v):
        return v % 2 == 1
    nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
    assert FPO.filter_out_dict(fn=is_odd, d=nums) == {'y':2,'r':4}

</section>

<details class="source"><summary>Source code</summary>

    def filter_out_dict(fn, d):
        '''
        ## FPO.filter_out_dict(...)
        The inverse of FPO.filterInObj(..), produces a new dictionary by calling a predicate function with each property value in the original dictionary. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new object. Otherwise, the value is included at the same property name.
        ### Arguments:
            fn:     predicate function; called with v (value), i (property name), and o (object) named arguments
            d:      dictionary to filter against
        ### Returns:
            dictionary
        ### Aliases:
            FPO.reject_dict(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = {'x':1,'y':2,'z':3,'r':4,'l':5}
            assert FPO.filter_out_dict(fn=is_odd, d=nums) == {'y':2,'r':4}
        '''
        r = {}
        for key,v in d.items():
            if fn(v) != True:
                r[key] = v
        return r

</details></dd>

<dt id="fpo.map_dict">`<span>def <span class="ident">map_dict</span></span>(<span>fn, d)</span>`</dt>

<dd>

<section class="desc">

## FPO.map_dict(…)

Produces a new dictionary by calling a mapper function with each property value in the original dictionary. The value the mapper function returns is inserted in the new object at that same property name. The new dictionary will always have the same number of properties as the original dictionary.

### Arguments:

    fn:     mapper function; called with v (value), i (index), and d (dictionary) named arguments
    d:   dictionary to-map against

### Returns:

    dictionary

### Example:

    def double(v, key): return v * 2
    nums = {'a': 1, 'b': 2, 'c': 3}
    assert FPO.map_dict(fn=double,d=nums) == {'a': 2, 'b': 4, 'c': 6}

</section>

<details class="source"><summary>Source code</summary>

    def map_dict(fn, d):
        '''
        ## FPO.map_dict(...)
        Produces a new dictionary by calling a mapper function with each property value in the original dictionary. The value the mapper function returns is inserted in the new object at that same property name. The new dictionary will always have the same number of properties as the original dictionary.
        ### Arguments:
            fn:     mapper function; called with v (value), i (index), and d (dictionary) named arguments
            d:   dictionary to-map against
        ### Returns:
            dictionary
        ### Example:
            def double(v, key): return v * 2
            nums = {'a': 1, 'b': 2, 'c': 3}
            assert FPO.map_dict(fn=double,d=nums) == {'a': 2, 'b': 4, 'c': 6}
        '''
        r = {}
        for key, v in d.items():
            r[key] = fn(v=v,key=key)
        return r

</details></dd>

<dt id="fpo.map_list">`<span>def <span class="ident">map_list</span></span>(<span>fn, l)</span>`</dt>

<dd>

<section class="desc">

## FPO.map_list(…)

Produces a new list by calling a mapper function with each value in the original list. The value the mapper function returns is inserted in the new list at that same position. The new list will always be the same length as the original list.

### Arguments:

    fn: mapper function; called with v (value) and l (list) named arguments
    l:  list to map against

### Returns:

    list

### Example:

    def double(v): return v * 2
    nums = [1,2,3]
    assert FPO.map_list(fn=double,l=nums) == [2,4,6]

</section>

<details class="source"><summary>Source code</summary>

    def map_list(fn, l):
        '''
        ## FPO.map_list(...)
        Produces a new list by calling a mapper function with each value in the original list. The value the mapper function returns is inserted in the new list at that same position. The new list will always be the same length as the original list.
        ### Arguments:
            fn: mapper function; called with v (value) and l (list) named arguments
            l:  list to map against
        ### Returns:
            list
        ### Example:
            def double(v): return v * 2
            nums = [1,2,3]
            assert FPO.map_list(fn=double,l=nums) == [2,4,6]
        '''
        r = []
        for v in l:
            r.append(fn(v=v))
        return r

</details></dd>

<dt id="fpo.memoise">`<span>def <span class="ident">memoise</span></span>(<span>fn, n=-1)</span>`</dt>

<dd>

<section class="desc">

## FPO.memoize(…)

For performance optimization reasons, wraps a function such that it remembers each set of arguments passed to it, associated with that underlying return value. If the wrapped function is called subsequent times with the same set of arguments, the cached return value is returned instead of being recomputed. Each wrapped function instance has its own separate cache, even if wrapping the same original function multiple times.

A set of arguments is "remembered" by being hashed to a string value to use as a cache key. This hashing is done internally with json.dumps(..), which is fast and works with many common value types. However, this hashing is by no means bullet-proof for all types, and does not guarantee collision-free. Use caution: generally, you should only use primitives (number, string, boolean, null, and None) or simple objects (dict, list) as arguments. If you use objects, always make sure to list properties in the same order to ensure proper hashing.

Unary functions (single argument; n of 1) with a primitive argument are the fastest for memoisation, so if possible, try to design functions that way. In these cases, specifying n as 1 will help ensure the best possible performance.

Warning: Be aware that if 1 is initially specified (or detected) for n, additional arguments later passed to the wrapped function are not considered in the memoisation hashing, though they will still be passed to the underlying function as-is. This may cause unexpected results (false-positives on cache hits); always make sure n matches the expected number of arguments.

### Arguments:

    fn: function to wrap
    n:  number of arguments to memoize; if omitted, tries to detect the arity (fn.length) to use.

### Returns:

    list

### Example:

    def sum(x,y):
        return x + y + random.randint(1,101)
    fa = FPO.memoise(fn=sum)
    fb = FPO.memoise(fn=sum, n=1)
    cached_a = fa(2,3)
    assert fa(2,3) == cached_a
    cached_b = fb(2,3)
    assert fb(2,4) == cached_b

</section>

<details class="source"><summary>Source code</summary>

    def memoise(fn,n=-1):
        '''
        ## FPO.memoize(...)
        For performance optimization reasons, wraps a function such that it remembers each set of arguments passed to it, associated with that underlying return value. If the wrapped function is called subsequent times with the same set of arguments, the cached return value is returned instead of being recomputed. Each wrapped function instance has its own separate cache, even if wrapping the same original function multiple times.

        A set of arguments is "remembered" by being hashed to a string value to use as a cache key. This hashing is done internally with json.dumps(..), which is fast and works with many common value types. However, this hashing is by no means bullet-proof for all types, and does not guarantee collision-free. Use caution: generally, you should only use primitives (number, string, boolean, null, and None) or simple objects (dict, list) as arguments. If you use objects, always make sure to list properties in the same order to ensure proper hashing.

        Unary functions (single argument; n of 1) with a primitive argument are the fastest for memoisation, so if possible, try to design functions that way. In these cases, specifying n as 1 will help ensure the best possible performance.

        Warning: Be aware that if 1 is initially specified (or detected) for n, additional arguments later passed to the wrapped function are not considered in the memoisation hashing, though they will still be passed to the underlying function as-is. This may cause unexpected results (false-positives on cache hits); always make sure n matches the expected number of arguments.
        ### Arguments:
            fn: function to wrap
            n:  number of arguments to memoize; if omitted, tries to detect the arity (fn.length) to use.
        ### Returns:
            list
        ### Example:
            def sum(x,y):
                return x + y + random.randint(1,101)
            fa = FPO.memoise(fn=sum)
            fb = FPO.memoise(fn=sum, n=1)
            cached_a = fa(2,3)
            assert fa(2,3) == cached_a
            cached_b = fb(2,3)
            assert fb(2,4) == cached_b
        '''
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

</details></dd>

<dt id="fpo.n_ary">`<span>def <span class="ident">n_ary</span></span>(<span>fn, props)</span>`</dt>

<dd>

<section class="desc">

## FPO.n_ary(…)

Wraps a function to restrict its inputs to only the named arguments as specified. It is similar to FPO.pluck.

### Arguments:

    fn:     function to wrap
    props:  list of property names to allow as named arguments; if empty, produces a "nullary" function -- won't receive any arguments.

### Returns:

    function

### Example:

    def foo(d): return d
    f = FPO.n_ary(fn=foo, props=['x','y','z'])
    assert f({'x': 1, 'y': 2, 'z': 3, 'w': 4}) == {'x': 1, 'y': 2, 'z': 3}

</section>

<details class="source"><summary>Source code</summary>

    def n_ary(fn,props):
        '''
        ## FPO.n_ary(...)
        Wraps a function to restrict its inputs to only the named arguments as specified. It is similar to FPO.pluck.
        ### Arguments:
            fn:     function to wrap
            props:  list of property names to allow as named arguments; if empty, produces a "nullary" function -- won't receive any arguments.
        ### Returns:
            function
        ### Example:
            def foo(d): return d
            f = FPO.n_ary(fn=foo, props=['x','y','z'])
            assert f({'x': 1, 'y': 2, 'z': 3, 'w': 4}) == {'x': 1, 'y': 2, 'z': 3}
        '''
        def n_aried(d):
            if bool(props) is not True:
                return fn()
            else:
                r = {}
                for key in props:
                    r[key] = d[key]
                return fn(r)
        return n_aried

</details></dd>

<dt id="fpo.partial">`<span>def <span class="ident">partial</span></span>(<span>fn, args)</span>`</dt>

<dd>

<section class="desc">

## FPO.partial(…)

Wraps a function with a new function that already has some of the arguments pre-specified, and is waiting for the rest of them on the next call. Unlike FPO.curry(..), you must specify all the remaining arguments on the next call of the partially-applied function.

With traditional FP libraries, partial(..) works in left-to-right order (as does FPO.std.partial(..)). That's why typically you also need a FPO.std.partialRight(..) if you want to partially-apply from the opposite direction.

However, using named arguments style – after all, that is the whole point of FPO! – order doesn't matter. For familiarity sake, FPO.partialRight(..) is provided, but it's just an alias to FPO.partial(..).

### Arguments:

    fn:     function to partially-apply
    args:   object containing the arguments to apply now

### Returns:

    function

### Example:

    def foo(x,y,z): return x + y + z
    f = FPO.partial(fn=foo, args={'x': 'a'});
    assert f(y='b', z='c') == 'abc'

</section>

<details class="source"><summary>Source code</summary>

    def partial(fn, args):
        '''
        ## FPO.partial(...)
        Wraps a function with a new function that already has some of the arguments pre-specified, and is waiting for the rest of them on the next call. Unlike FPO.curry(..), you must specify all the remaining arguments on the next call of the partially-applied function.

        With traditional FP libraries, partial(..) works in left-to-right order (as does FPO.std.partial(..)). That's why typically you also need a FPO.std.partialRight(..) if you want to partially-apply from the opposite direction.

        However, using named arguments style -- after all, that is the whole point of FPO! -- order doesn't matter. For familiarity sake, FPO.partialRight(..) is provided, but it's just an alias to FPO.partial(..).
        ### Arguments:
            fn:     function to partially-apply
            args:   object containing the arguments to apply now
        ### Returns:
            function
        ### Example:
            def foo(x,y,z): return x + y + z
            f = FPO.partial(fn=foo, args={'x': 'a'});
            assert f(y='b', z='c') == 'abc'
        '''
        def partialed(**kwargs):
            l_kwargs = copy.copy(kwargs)
            l_kwargs.update(args)
            return fn(**l_kwargs)
        return partialed

</details></dd>

<dt id="fpo.pick">`<span>def <span class="ident">pick</span></span>(<span>d, props)</span>`</dt>

<dd>

<section class="desc">

## FPO.pick(…)

Returns a new dictionary with only the specified properties from the original dictionary. Includes only properties from the original dictionary.

### Arguments:

    d:      dictionary to pick properties from
    props:  list of property names to pick from the object; if a property does not exist on the original dictionary, it is not added to the new dictionary, unlike FPO.pickAll(..).

### Returns:

    dictionary

### Example:

    d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.pick(d,props=['x','y']) == {'x': 1, 'y': 2}

</section>

<details class="source"><summary>Source code</summary>

    def pick(d,props):
        '''
        ## FPO.pick(...)
        Returns a new dictionary with only the specified properties from the original dictionary. Includes only properties from the original dictionary.
        ### Arguments:
            d:      dictionary to pick properties from
            props:  list of property names to pick from the object; if a property does not exist on the original dictionary, it is not added to the new dictionary, unlike FPO.pickAll(..).
        ### Returns:
            dictionary
        ### Example:
            d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
            assert FPO.pick(d,props=['x','y']) == {'x': 1, 'y': 2}
        '''
        r = {}
        for i in props:
            if i in d:
                r[i] = d[i]
        return r

</details></dd>

<dt id="fpo.pick_all">`<span>def <span class="ident">pick_all</span></span>(<span>d, props)</span>`</dt>

<dd>

<section class="desc">

## FPO.pick_all(…)

Returns a new dictionary with only the specified properties from the original dictionary. Includes all specified properties.

### Arguments:

    d:      dictionary to pick properties from
    props:  list of property names to pick from the dictionary; even if a property does not exist on the original dictionary, it is still added to the new object with an undefined value, unlike FPO.pick(..).

### Returns:

    dictionary

### Example:

    d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.pick_all(d,props=['x','y','r']) == {'x': 1, 'y': 2, 'r': None}

</section>

<details class="source"><summary>Source code</summary>

    def pick_all(d, props):
        '''
        ## FPO.pick_all(...)
        Returns a new dictionary with only the specified properties from the original dictionary. Includes all specified properties.
        ### Arguments:
            d:      dictionary to pick properties from
            props:  list of property names to pick from the dictionary; even if a property does not exist on the original dictionary, it is still added to the new object with an undefined value, unlike FPO.pick(..).
        ### Returns:
            dictionary
        ### Example:
            d = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
            assert FPO.pick_all(d,props=['x','y','r']) == {'x': 1, 'y': 2, 'r': None}
        '''
        r = {}
        for i in props:
            if i in d:
                r[i] = d[i]
            else:
                r[i] = None
        return r

</details></dd>

<dt id="fpo.pipe">`<span>def <span class="ident">pipe</span></span>(<span>fns)</span>`</dt>

<dd>

<section class="desc">

## FPO.pipe(…)

Produces a new function that's the composition of a list of functions. Functions are composed left-to-right (unlike FPO.compose(..)) from the array.

### Arguments:

    fns:    list of functions

### Returns:

    function

### Example:

    f = FPO.pipe([
        lambda v: v+2,
        lambda v: v*2,
        lambda v: v-2,
    ])
    assert f(10) == 22

</section>

<details class="source"><summary>Source code</summary>

    def pipe(fns):
        '''
        ## FPO.pipe(...)
        Produces a new function that's the composition of a list of functions. Functions are composed left-to-right (unlike FPO.compose(..)) from the array.
        ### Arguments:
            fns:    list of functions
        ### Returns:
            function
        ### Example:
            f = FPO.pipe([
                lambda v: v+2,
                lambda v: v*2,
                lambda v: v-2,
            ])
            assert f(10) == 22
        '''
        def piped(v):
            result = v
            for fn in fns:
                result = fn(v=result)
            return result
        return piped

</details></dd>

<dt id="fpo.pluck">`<span>def <span class="ident">pluck</span></span>(<span>l, *args)</span>`</dt>

<dd>

<section class="desc">

## FPO.pluck(…)

Plucks properties form the given list and return a list of properties' values

### Arguments:

    l:   list
    *args:  properties

### Returns:

    a list of values

### Example:

    l = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
    assert FPO.pluck(l, 'x', 'y') == [[1, 2], [3, 4]]
    assert FPO.pluck(l, 'x') == [1, 3]

</section>

<details class="source"><summary>Source code</summary>

    def pluck(l, *args):
        '''
        ## FPO.pluck(...)
        Plucks properties form the given list and return a list of properties' values
        ### Arguments:
            l:   list
            *args:  properties
        ### Returns:
            a list of values
        ### Example:
            l = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
            assert FPO.pluck(l, 'x', 'y') == [[1, 2], [3, 4]]
            assert FPO.pluck(l, 'x') == [1, 3]
        '''
        fn = lambda d, *args: [d[arg] for arg in args]
        r = [fn(o, *args) for o in l]
        if len(args) == 1:
            return [v[0] for v in r]
        else:
            return r

</details></dd>

<dt id="fpo.prop">`<span>def <span class="ident">prop</span></span>(<span>d, prop)</span>`</dt>

<dd>

<section class="desc">

## FPO.prop(…)

Extracts a property's value from a dictionary.

### Arguments:

    d:    dictionary to pull the property value from
    prop: property name to pull from the dictionary

### Returns:

    any

### Example:

    obj = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert FPO.prop(d=obj, prop='y') == 2

</section>

<details class="source"><summary>Source code</summary>

    def prop(d,prop):
        '''
        ## FPO.prop(...)
        Extracts a property's value from a dictionary.
        ### Arguments:
            d:    dictionary to pull the property value from
            prop: property name to pull from the dictionary
        ### Returns:
            any
        ### Example:
            obj = {'x': 1, 'y': 2, 'z': 3, 'w': 4}
            assert FPO.prop(d=obj, prop='y') == 2
        '''
        return d[prop]

</details></dd>

<dt id="fpo.reassoc">`<span>def <span class="ident">reassoc</span></span>(<span>d, props)</span>`</dt>

<dd>

<section class="desc">

## FPO.reassoc(…)

Like a mixture between FPO.pick(..) and FPO.setProp(..), creates a new dictionary that has properties remapped from original names to new names. Any properties present on the original dictionary that aren't remapped are copied with the same name.

### Arguments:

    d:      dictionary to remap properties from
    props:  dictionary whose key/value pairs are sourceProp: targetProp remappings

### Returns:

    dictionary

### Example:

    obj = dict(zip(['x','y','z'],[1, 2, 3]))
    assert FPO.reassoc(d=obj, props={'x': 'a', 'y': 'b'}) == {'a': 1, 'b': 2, 'z': 3}
    assert obj == {'x': 1, 'y': 2, 'z': 3}

</section>

<details class="source"><summary>Source code</summary>

    def reassoc(d,props):
        '''
        ## FPO.reassoc(...)
        Like a mixture between FPO.pick(..) and FPO.setProp(..), creates a new dictionary that has properties remapped from original names to new names. Any properties present on the original dictionary that aren't remapped are copied with the same name.
        ### Arguments:
            d:      dictionary to remap properties from
            props:  dictionary whose key/value pairs are sourceProp: targetProp remappings
        ### Returns:
            dictionary
        ### Example:
            obj = dict(zip(['x','y','z'],[1, 2, 3]))
            assert FPO.reassoc(d=obj, props={'x': 'a', 'y': 'b'}) == {'a': 1, 'b': 2, 'z': 3}
            assert obj == {'x': 1, 'y': 2, 'z': 3}
        '''
        r = {}
        for k,v in d.items():
            if k in props:
                r[props[k]] = d[k]
            else:
                r[k] = d[k]
        return r

</details></dd>

<dt id="fpo.reduce">`<span>def <span class="ident">reduce</span></span>(<span>fn, l=[], v=None)</span>`</dt>

<dd>

<section class="desc">

## FPO.reduce(..)

Processes a list from left-to-right (unlike FPO.reduceRight(..)), successively combining (aka "reducing", "folding") two values into one, until the entire list has been reduced to a single value. An initial value for the reduction can optionally be provided.

### Arguments:

    fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
    l:  list to reduce
    v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.

### Returns:

    any

### Example:

    def str_concat(acc,v):
        return acc + v
    vowels = ["a","e","i","o","u","y"]
    assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
    assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
    assert vowels == ["a","e","i","o","u","y"]

</section>

<details class="source"><summary>Source code</summary>

    def reduce(fn,l=[],v=None):
        '''
        ## FPO.reduce(..)
        Processes a list from left-to-right (unlike FPO.reduceRight(..)), successively combining (aka "reducing", "folding") two values into one, until the entire list has been reduced to a single value. An initial value for the reduction can optionally be provided.
        ### Arguments:
            fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
            l:  list to reduce
            v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
        ### Returns:
            any
        ### Example:
            def str_concat(acc,v):
                return acc + v
            vowels = ["a","e","i","o","u","y"]
            assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
            assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
            assert vowels == ["a","e","i","o","u","y"]
        '''
        orig_l = l
        initial_v = v
        if initial_v is None and len(l) > 0:
            initial_v = l[0]
            l = l[1:]
        for e in l:
            initial_v = fn(acc=initial_v, v=e)
        return initial_v

</details></dd>

<dt id="fpo.reduce_dict">`<span>def <span class="ident">reduce_dict</span></span>(<span>fn, d, v=None)</span>`</dt>

<dd>

<section class="desc">

## FPO.reduce_dict(..)

Processes an dictionary's properties (in enumeration order), successively combining (aka "reducing", "folding") two values into one, until all the dictionary's properties have been reduced to a single value. An initial value for the reduction can optionally be provided.

### Arguments:

    fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
    d:  dictionary to reduce
    v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.

### Returns:

    any

### Example:

    def str_concat(acc,v):
        return acc + v
    vowels = ["a","e","i","o","u","y"]
    assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
    assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
    assert vowels == ["a","e","i","o","u","y"]

</section>

<details class="source"><summary>Source code</summary>

    def reduce_dict(fn,d,v=None):
        '''
        ## FPO.reduce_dict(..)
        Processes an dictionary's properties (in enumeration order), successively combining (aka "reducing", "folding") two values into one, until all the dictionary's properties have been reduced to a single value. An initial value for the reduction can optionally be provided.
        ### Arguments:
            fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
            d:  dictionary to reduce
            v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
        ### Returns:
            any
        ### Example:
            def str_concat(acc,v):
                return acc + v
            vowels = ["a","e","i","o","u","y"]
            assert FPO.reduce(fn=str_concat, l=vowels) == 'aeiouy'
            assert FPO.reduce(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: aeiouy'
            assert vowels == ["a","e","i","o","u","y"]
        '''
        init_k = next(iter(d))
        r = d[init_k]
        for key,value in d.items():
            if key is not init_k:
                r = fn(acc=r, v=value)
        if bool(v) is True:
            return v + r
        return r

</details></dd>

<dt id="fpo.reduce_right">`<span>def <span class="ident">reduce_right</span></span>(<span>fn, l, v=None)</span>`</dt>

<dd>

<section class="desc">

## FPO.reduce_right(..)

Processes a list from right-to-left (unlike FPO.reduce(..)), successively combining (aka "reducing", "folding") two values into one, until the entire list has been reduced to a single value. An initial value for the reduction can optionally be provided. If the array is empty, the initial value is returned (or undefined if it was omitted).

### Arguments:

    fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
    l:  list to reduce
    v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.

### Returns:

    any

### Example:

    def str_concat(acc,v):
        return acc + v
    vowels = ["a","e","i","o","u","y"]
    assert FPO.reduce_right(fn=str_concat, l=vowels) == 'yuoiea'
    assert FPO.reduce_right(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: yuoiea'
    assert vowels == ["a","e","i","o","u","y"]

</section>

<details class="source"><summary>Source code</summary>

    def reduce_right(fn,l,v=None):
        '''
        ## FPO.reduce_right(..)
        Processes a list from right-to-left (unlike FPO.reduce(..)), successively combining (aka "reducing", "folding") two values into one, until the entire list has been reduced to a single value.
        An initial value for the reduction can optionally be provided. If the array is empty, the initial value is returned (or undefined if it was omitted).
        ### Arguments:
            fn: reducer function; called with acc (acculumator), v (value) and l (list) named arguments
            l:  list to reduce
            v:  (optional) initial value to use for the reduction; if provided, the first reduction will pass to the reducer the initial value as the acc and the first value from the array as v. Otherwise, the first reduction has the first value of the array as acc and the second value of the array as v.
        ### Returns:
            any
        ### Example:
            def str_concat(acc,v):
                return acc + v
            vowels = ["a","e","i","o","u","y"]
            assert FPO.reduce_right(fn=str_concat, l=vowels) == 'yuoiea'
            assert FPO.reduce_right(fn=str_concat, l=vowels, v='vowels: ') == 'vowels: yuoiea'
            assert vowels == ["a","e","i","o","u","y"]
        '''
        rl = l[::-1]
        r = rl[0]
        for e in rl[1:]:
            r = fn(acc=r, v=e)
        if bool(v) is True:
            return v + r
        return r

</details></dd>

<dt id="fpo.reject">`<span>def <span class="ident">reject</span></span>(<span>fn, l)</span>`</dt>

<dd>

<section class="desc">

## FPO.filter_out(…)

The inverse of FPO.filterIn(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new list. Otherwise, the value is included.

### Arguments:

    fn:     predicate function; called with v (value), i (index), and l (list) named arguments
    l:    list to filter against

### Returns:

    list

### Aliases:

    FPO.reject(..)

### Example:

    def is_odd(v):
        return v % 2 == 1
    nums = [1,2,3,4,5]
    assert FPO.filter_out(fn=is_odd, l=nums) == [2,4]

</section>

<details class="source"><summary>Source code</summary>

    def filter_out(fn,l):
        '''
        ## FPO.filter_out(...)
        The inverse of FPO.filterIn(..), produces a new list by calling a predicate function with each value in the original list. For each value, if the predicate function returns true (or truthy), the value is omitted from (aka, filtered out of) the new list. Otherwise, the value is included.
        ### Arguments:
            fn:     predicate function; called with v (value), i (index), and l (list) named arguments
            l:    list to filter against
        ### Returns:
            list
        ### Aliases:
            FPO.reject(..)
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            nums = [1,2,3,4,5]
            assert FPO.filter_out(fn=is_odd, l=nums) == [2,4]
        '''
        r = []
        for e in l:
            if fn(e) is not True:
               r.append(e)
        return r

</details></dd>

<dt id="fpo.remap">`<span>def <span class="ident">remap</span></span>(<span>fn, args)</span>`</dt>

<dd>

<section class="desc">

## FPO.remap(..)

Remaps the expected named arguments of a function. This is useful to adapt a function to be used if the arguments passed in will be different than what the function expects. A common usecase will be to adapt a function so it's suitable for use as a mapper/predicate/reducer function, or for composition.

### Arguments:

    fn:     function to remap
    args:   dictionary whose key/value pairs represent the origArgName: newArgName mappings

### Returns:

    function

### Example:

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

</section>

<details class="source"><summary>Source code</summary>

    def remap(fn, args):
        '''
        ## FPO.remap(..)
        Remaps the expected named arguments of a function. This is useful to adapt a function to be used if the arguments passed in will be different than what the function expects.
        A common usecase will be to adapt a function so it's suitable for use as a mapper/predicate/reducer function, or for composition.
        ### Arguments:
            fn:     function to remap
            args:   dictionary whose key/value pairs represent the origArgName: newArgName mappings
        ### Returns:
            function
        ### Example:
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
        def remaped(**kwargs):
            l_kwargs = reassoc(kwargs,props=args)
            return fn(**l_kwargs)
        return remaped

</details></dd>

<dt id="fpo.set_prop">`<span>def <span class="ident">set_prop</span></span>(<span>d, prop, v)</span>`</dt>

<dd>

<section class="desc">

## FPO.set_prop(…)

Creates a shallow clone of a dictionary, assigning the specified property value to the new dictionary.

### Arguments:

    d:      (optional) object to clone; if omitted, defaults to a new empty dictionary
    prop:   property name where to set the value on the new dictionary
    v:      value

### Returns:

    any

### Example:

    obj = dict(x=1, y=2,z=3)
    assert FPO.set_prop(d=obj, prop='w', v=4) == {'x': 1, 'y': 2, 'z': 3, 'w': 4}
    assert obj == {'x': 1, 'y': 2, 'z': 3}

</section>

<details class="source"><summary>Source code</summary>

    def set_prop(d,prop,v):
        '''
        ##FPO.set_prop(...)
        Creates a shallow clone of a dictionary, assigning the specified property value to the new dictionary.
        ### Arguments:
            d:      (optional) object to clone; if omitted, defaults to a new empty dictionary
            prop:   property name where to set the value on the new dictionary
            v:      value
        ### Returns:
            any
        ### Example:
            obj = dict(x=1, y=2,z=3)
            assert FPO.set_prop(d=obj, prop='w', v=4) == {'x': 1, 'y': 2, 'z': 3, 'w': 4}
            assert obj == {'x': 1, 'y': 2, 'z': 3}
        '''
        if bool(d) is True:
            r = copy.copy(d)
        else:
            r = {}
        r[prop] = v
        return r

</details></dd>

<dt id="fpo.tail">`<span>def <span class="ident">tail</span></span>(<span>v)</span>`</dt>

<dd>

<section class="desc">

## FPO.tail(…)

Returns everything else in the value except the element as accessed at index 0; basically the inverse of FPO.head(..)

### Arguments:

    v:   list/string/dictionary

### Returns:

    any

### Example:

    assert FPO.tail(v={'a':42,'b':56,'c':34}) == {'b':56,'c':34}
    assert FPO.tail(v=[1,2,3,4]) == [2,3,4]
    assert FPO.tail(v=(42,56,32)) == (56,32)
    assert FPO.tail(v='abc') == 'bc'
    assert FPO.tail(v=[]) == None
    assert FPO.tail(v={}) == None
    assert FPO.tail(v='') == None

</section>

<details class="source"><summary>Source code</summary>

    def tail(v):
        '''
        ## FPO.tail(...)
        Returns everything else in the value except the element as accessed at index 0; basically the inverse of FPO.head(..)
        ### Arguments:
            v:   list/string/dictionary
        ### Returns:
            any
        ### Example:
            assert FPO.tail(v={'a':42,'b':56,'c':34}) == {'b':56,'c':34}
            assert FPO.tail(v=[1,2,3,4]) == [2,3,4]
            assert FPO.tail(v=(42,56,32)) == (56,32)
            assert FPO.tail(v='abc') == 'bc'
            assert FPO.tail(v=[]) == None
            assert FPO.tail(v={}) == None
            assert FPO.tail(v='') == None
        '''
        if bool(v) is not True:
            return None
        elif isinstance(v, dict) is True:
            init_k = next(iter(v))
            r = {}
            for key,value in v.items():
                if key is not init_k:
                    r[key] = value
            return r
        elif isinstance(v, (list, tuple)) is True:
            return v[1:]
        elif isinstance(v, str) is True:
            return v[1:]

</details></dd>

<dt id="fpo.take">`<span>def <span class="ident">take</span></span>(<span>iterable, n=1)</span>`</dt>

<dd>

<section class="desc">

## FPO.take(…)

Returns the specified number of elements from the value, starting from the beginning.

### Arguments:

    iterable:   list/string
    n:          number of elements to take from the beginning of the value; if omitted, defaults to `1`

### Returns:

    list/string

### Example:

    items = [2,4,6,8,10]
    assert FPO.take(items, 3) == [2,4,6]
    assert FPO.take(items) == [2]
    assert FPO.take({'apple','banana','cherry'}, 2) == ['apple','banana']

</section>

<details class="source"><summary>Source code</summary>

    def take(iterable, n=1):
        '''
        ## FPO.take(...)
        Returns the specified number of elements from the value, starting from the beginning.
        ### Arguments:
            iterable:   list/string
            n:          number of elements to take from the beginning of the value; if omitted, defaults to `1`
        ### Returns:
            list/string
        ### Example:
            items = [2,4,6,8,10]
            assert FPO.take(items, 3) == [2,4,6]
            assert FPO.take(items) == [2]
            assert FPO.take({'apple','banana','cherry'}, 2) == ['apple','banana']
        '''
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

</details></dd>

<dt id="fpo.trampoline">`<span>def <span class="ident">trampoline</span></span>(<span>fn)</span>`</dt>

<dd>

<section class="desc">

## FPO.trampoline(…)

Wraps a continuation-returning recursive function in another function that will run it until it no longer returns another continuation function. Trampolines are an alternative to tail calls.

### Arguments:

    fn:     function to run

### Returns:

    function

### Example:

    def sum(total,x):
        if x <= 1:
        return total + x
    return lambda : sum(total+x, x-1)
    assert FPO.trampoline(fn=sum)(0,5) == 15

</section>

<details class="source"><summary>Source code</summary>

    def trampoline(fn):
        '''
        ## FPO.trampoline(...)
        Wraps a continuation-returning recursive function in another function that will run it until it no longer returns another continuation function. Trampolines are an alternative to tail calls.
        ### Arguments:
            fn:     function to run
        ### Returns:
            function
        ### Example:
            def sum(total,x):
                if x <= 1:
                return total + x
            return lambda : sum(total+x, x-1)
            assert FPO.trampoline(fn=sum)(0,5) == 15
        '''
        def trampolined(*args, **kwargs):
            if bool(args):
                r = fn(*args)
            else:
                r = fn(**kwargs)
            while callable(r) is True:
                r = r()
            return r
        return trampolined

</details></dd>

<dt id="fpo.transduce_fn">`<span>def <span class="ident">transduce_fn</span></span>(<span>fn, co, v, l=[])</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_transduce(…)

Produces a reducer from a specified transducer and combination function. Then runs a reduction on a list, using that reducer, starting with the specified initial value. Note: When composing transducers, the effective order of operations is reversed from normal composition. Instead of expecting composition to be right-to-left, the effective order will be left-to-right (see below).

### Arguments:

    fn: transducer function
    co: combination function for the transducer
    v: initial value for the reduction
    l: the list for the reduction

### Returns:

    any

### Example:

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

</section>

<details class="source"><summary>Source code</summary>

    def transduce_fn(fn,co,v,l=[]):
        '''
        ## FPO.transducer_transduce(...)
        Produces a reducer from a specified transducer and combination function. Then runs a reduction on a list, using that reducer, starting with the specified initial value.
        Note: When composing transducers, the effective order of operations is reversed from normal composition. Instead of expecting composition to be right-to-left, the effective order will be left-to-right (see below).
        ### Arguments:
            fn: transducer function
            co: combination function for the transducer
            v: initial value for the reduction
            l: the list for the reduction
        ### Returns:
            any
        ### Example:
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
        '''
        transducer = fn
        combination_fn = co
        initial_value = v
        reducer = transducer(v=combination_fn)
        return reduce(fn=reducer, v=initial_value, l=l)

</details></dd>

<dt id="fpo.transducer_bool_and">`<span>def <span class="ident">transducer_bool_and</span></span>(<span>acc, v)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_bool_and(…)

A reducer function. For transducing purposes, a combination function that takes two booleans and ANDs them together. The result is the logical AND of the two values.

### Arguments:

    acc:    acculumator
    v:  value

### Returns:

    true/false

### Example:

    assert FPO.transducer_bool_and(acc=True, v=True) == True
    assert FPO.transducer_bool_and(acc=False, v=True) == False

</section>

<details class="source"><summary>Source code</summary>

    def transducer_bool_and(acc,v):
        '''
        ## FPO.transducer_bool_and(...)
        A reducer function. For transducing purposes, a combination function that takes two booleans and ANDs them together. The result is the logical AND of the two values.
        ### Arguments:
            acc:    acculumator
            v:  value
        ### Returns:
            true/false
        ### Example:
            assert FPO.transducer_bool_and(acc=True, v=True) == True
            assert FPO.transducer_bool_and(acc=False, v=True) == False
        '''
        if bool(acc) and bool(v) is True:
            return True
        else:
            return False

</details></dd>

<dt id="fpo.transducer_bool_or">`<span>def <span class="ident">transducer_bool_or</span></span>(<span>acc, v)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_bool_or(…)

A reducer function. For transducing purposes, a combination function that takes two booleans and ORs them together. The result is the logical OR of the two values.

### Arguments:

    acc:    acculumator
    v:  value

### Returns:

    true/false

### Example:

    assert FPO.transducer_bool_or(acc=True, v=True) == True
    assert FPO.transducer_bool_or(acc=False, v=False) == False
    assert FPO.transducer_bool_or(acc=False, v=True) == True

</section>

<details class="source"><summary>Source code</summary>

    def transducer_bool_or(acc,v):
        '''
        ## FPO.transducer_bool_or(...)
        A reducer function. For transducing purposes, a combination function that takes two booleans and ORs them together. The result is the logical OR of the two values.
        ### Arguments:
            acc:    acculumator
            v:  value
        ### Returns:
            true/false
        ### Example:
            assert FPO.transducer_bool_or(acc=True, v=True) == True
            assert FPO.transducer_bool_or(acc=False, v=False) == False
            assert FPO.transducer_bool_or(acc=False, v=True) == True
        '''
        if bool(acc) or bool(v) is True:
            return True
        else:
            return False

</details></dd>

<dt id="fpo.transducer_default">`<span>def <span class="ident">transducer_default</span></span>(<span>acc, v)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_default(…)

A reducer function. For transducing purposes, a combination function that's a default placeholder. It returns only the acc that's passed to it. The behavior here is almost the same as FPO.identity(..), except that returns acc instead of v.

### Arguments:

    acc:    acculumator
    v:  value

### Returns:

    any

### Example:

    assert FPO.transducer_default(acc=3, v=1) == 3

</section>

<details class="source"><summary>Source code</summary>

    def transducer_default(acc,v):
        '''
        ## FPO.transducer_default(...)
        A reducer function. For transducing purposes, a combination function that's a default placeholder. It returns only the acc that's passed to it. The behavior here is almost the same as FPO.identity(..), except that returns acc instead of v.
        ### Arguments:
            acc:    acculumator
            v:  value
        ### Returns:
            any
        ### Example:
            assert FPO.transducer_default(acc=3, v=1) == 3
        '''
        return acc

</details></dd>

<dt id="fpo.transducer_filter">`<span>def <span class="ident">transducer_filter</span></span>(<span>*args, **kwargs)</span>`</dt>

<dd><details class="source"><summary>Source code</summary>

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

</details></dd>

<dt id="fpo.transducer_filter_fn">`<span>def <span class="ident">transducer_filter_fn</span></span>(<span>fn, v=None)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_filter(…)

For transducing purposes, wraps a predicate function as a filter-transducer. Typically, this filter-transducer is then composed with other filter-transducers and/or map-transducers. The resulting transducer is then passed to FPO.transducers.transduce(..).

### Arguments:

    fn:    predicate function

### Returns:

    function

### Example:

    def is_odd(v):
        return v % 2 == 1
    def list_push(acc, v):
        acc.append(v)
        return acc
    nums = [1,2,3,4,5]
    filter_transducer = FPO.transducer_filter(fn=is_odd)
    r = FPO.transducer_transduce(fn=filter_transducer, co=list_push, v=[], l=nums)
    assert r == [1,3,5]

</section>

<details class="source"><summary>Source code</summary>

    def transducer_filter_fn(fn,v=None):
        '''
        ## FPO.transducer_filter(...)
        For transducing purposes, wraps a predicate function as a filter-transducer. Typically, this filter-transducer is then composed with other filter-transducers and/or map-transducers. The resulting transducer is then passed to FPO.transducers.transduce(..).
        ### Arguments:
            fn:    predicate function
        ### Returns:
            function
        ### Example:
            def is_odd(v):
                return v % 2 == 1
            def list_push(acc, v):
                acc.append(v)
                return acc
            nums = [1,2,3,4,5]
            filter_transducer = FPO.transducer_filter(fn=is_odd)
            r = FPO.transducer_transduce(fn=filter_transducer, co=list_push, v=[], l=nums)
            assert r == [1,3,5]
        '''
        predicated_fn = fn
        combination_fn = v
        #till waiting on the combination function?
        if combination_fn is None:
            #Note: the combination function is usually a composed
            #function, so we expect the argument by itself,
            #not wrapped in a dictionary
            def curried(v):
                nonlocal predicated_fn
                return transducer_filter_fn(fn=predicated_fn,v=v)
            return curried

        def reducer(acc,v):
            nonlocal predicated_fn, combination_fn
            if predicated_fn(v):
                return combination_fn(acc, v)
            return acc
        return reducer

</details></dd>

<dt id="fpo.transducer_into">`<span>def <span class="ident">transducer_into</span></span>(<span>*args, **kwargs)</span>`</dt>

<dd><details class="source"><summary>Source code</summary>

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

</details></dd>

<dt id="fpo.transducer_into_fn">`<span>def <span class="ident">transducer_into_fn</span></span>(<span>fn, v, l)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_into(…)

Selects an appropriate combination function (reducer) based on the provided initial value. Then runs FPO.transducers.transduce(..) under the covers.

Detects initial values of boolean, number, string, and list types, and dispatches to the appropriate combination function accordingly (FPO.transducers.number(..), etc). Note: A boolean initial value selects FPO.transducer_bool_and(..).

Note: When composing transducers, the effective order of operations is reversed from normal composition. Instead of expecting composition to be right-to-left, the effective order will be left-to-right (see below).

### Arguments:

    fn: transducer function
    v:  initial value for the reduction; also used to select the appropriate combination function (reducer) for the transducing.
    l: the list for the reductiontransduce_fn

### Example:

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

</section>

<details class="source"><summary>Source code</summary>

    def transducer_into_fn(fn,v,l):
        '''
        ## FPO.transducer_into(...)
        Selects an appropriate combination function (reducer) based on the provided initial value. Then runs FPO.transducers.transduce(..) under the covers.

        Detects initial values of boolean, number, string, and list types, and dispatches to the appropriate combination function accordingly (FPO.transducers.number(..), etc). Note: A boolean initial value selects FPO.transducer_bool_and(..).

        Note: When composing transducers, the effective order of operations is reversed from normal composition. Instead of expecting composition to be right-to-left, the effective order will be left-to-right (see below).
        ### Arguments:
            fn: transducer function
            v:  initial value for the reduction; also used to select the appropriate combination function (reducer) for the transducing.
            l: the list for the reductiontransduce_fn
        ### Example:
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
        '''
        transducer = fn
        combination_fn = transducer_default
        if isinstance(v, bool):
            combination_fn = transducer_bool_and
        elif isinstance(v, str):
            combination_fn = transducer_string
        elif isinstance(v, int):
            combination_fn = transducer_number
        elif isinstance(v, list):
            combination_fn = transducer_list
        else:
            transducer_default
        return transduce_fn(fn=transducer, co=combination_fn, v=v, l=l)

</details></dd>

<dt id="fpo.transducer_list">`<span>def <span class="ident">transducer_list</span></span>(<span>acc, v)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_list(…)

A reducer function. For transducing purposes, a combination function that takes an array and a value, and mutates the array by pushing the value onto the end of it. The mutated array is returned. _This function has side-effects_, for performance reasons. It should be used with caution.

### Arguments:

    acc:    acculumator
    v:  value

### Returns:

    list

### Example:

    arr = [1,2,3]
    FPO.transducer_list(acc=arr,v=4)
    assert arr == [1,2,3,4]

</section>

<details class="source"><summary>Source code</summary>

    def transducer_list(acc,v):
        '''
        ## FPO.transducer_list(...)
        A reducer function. For transducing purposes, a combination function that takes an array and a value, and mutates the array by pushing the value onto the end of it. The mutated array is returned.
        *This function has side-effects*, for performance reasons. It should be used with caution.
        ### Arguments:
            acc:    acculumator
            v:  value
        ### Returns:
            list
        ### Example:
            arr = [1,2,3]
            FPO.transducer_list(acc=arr,v=4)
            assert arr == [1,2,3,4]
        '''
        acc.append(v)
        return acc

</details></dd>

<dt id="fpo.transducer_map">`<span>def <span class="ident">transducer_map</span></span>(<span>*args, **kwargs)</span>`</dt>

<dd><details class="source"><summary>Source code</summary>

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

</details></dd>

<dt id="fpo.transducer_map_fn">`<span>def <span class="ident">transducer_map_fn</span></span>(<span>fn, v=None)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_map(…)

For transducing purposes, wraps a mapper function as a map-transducer. Typically, this map-transducer is then composed with other filter-transducers and/or map-transducers. The resulting transducer is then passed to FPO.transducers.transduce(..). The map-transducer is not a reducer itself; it's expecting a combination function (reducer), which will then produce a filter-reducer. So alternately, you can manually create the map-reducer and use it directly with a regular FPO.reduce(..) reduction.

### Arguments:

    fn: mapper function

### Returns:

    function

### Example:

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

</section>

<details class="source"><summary>Source code</summary>

    def transducer_map_fn(fn,v=None):
        '''
        ## FPO.transducer_map(...)
        For transducing purposes, wraps a mapper function as a map-transducer. Typically, this map-transducer is then composed with other filter-transducers and/or map-transducers. The resulting transducer is then passed to FPO.transducers.transduce(..).
        The map-transducer is not a reducer itself; it's expecting a combination function (reducer), which will then produce a filter-reducer. So alternately, you can manually create the map-reducer and use it directly with a regular FPO.reduce(..) reduction.
        ### Arguments:
            fn: mapper function
        ### Returns:
            function
        ### Example:
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
        '''
        mapper_fn = fn
        combination_fn = v
        #till waiting on the combination function?
        if combination_fn is None:
            #Note: the combination function is usually a composed
            #function, so we expect the argument by itself,
            #not wrapped in a dictionary
            def curried(v):
                nonlocal mapper_fn
                return transducer_map_fn(fn=mapper_fn,v=v)
            return curried

        def reducer(acc,v):
            nonlocal mapper_fn, combination_fn
            return combination_fn(acc,v=mapper_fn(v))
        return reducer

</details></dd>

<dt id="fpo.transducer_number">`<span>def <span class="ident">transducer_number</span></span>(<span>acc, v)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_number(…)

A reducer function. For transducing purposes, a combination function that adds together the two numbers passed into it. The result is the sum.

### Arguments:

    acc: acculumator
    v: value

### Returns:

    number

### Example:

    assert FPO.transducer_number( acc=3, v=4) == 7

</section>

<details class="source"><summary>Source code</summary>

    def transducer_number(acc,v):
        '''
        ## FPO.transducer_number(...)
        A reducer function. For transducing purposes, a combination function that adds together the two numbers passed into it. The result is the sum.
        ### Arguments:
            acc: acculumator
            v: value
        ### Returns:
            number
        ### Example:
            assert FPO.transducer_number( acc=3, v=4) == 7
        '''
        return acc + v

</details></dd>

<dt id="fpo.transducer_string">`<span>def <span class="ident">transducer_string</span></span>(<span>acc, v)</span>`</dt>

<dd>

<section class="desc">

## FPO.transducer_string(…)

A reducer function. For transducing purposes, a combination function that concats the two strings passed into it. The result is the concatenation.

### Arguments:

    acc: acculumator
    v: value

### Returns:

    string

### Example:

    assert FPO.transducer_string( acc='hello', v='world') == 'helloworld'

</section>

<details class="source"><summary>Source code</summary>

    def transducer_string(acc,v):
        '''
        ## FPO.transducer_string(...)
        A reducer function. For transducing purposes, a combination function that concats the two strings passed into it. The result is the concatenation.
        ### Arguments:
            acc: acculumator
            v: value
        ### Returns:
            string
        ### Example:
            assert FPO.transducer_string( acc='hello', v='world') == 'helloworld'
        '''
        return str(acc) + str(v)

</details></dd>

<dt id="fpo.transducer_transduce">`<span>def <span class="ident">transducer_transduce</span></span>(<span>*args, **kwargs)</span>`</dt>

<dd><details class="source"><summary>Source code</summary>

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

</details></dd>

<dt id="fpo.unapply">`<span>def <span class="ident">unapply</span></span>(<span>fn, props)</span>`</dt>

<dd>

<section class="desc">

## FPO.unapply(..)

Wraps a function to gather individual positional arguments into an object argument.

### Arguments:

    fn:     function to wrap
    props:  list of property names (strings) to indicate the order to gather individual positional arguments as properties.

### Returns:

    function

## Example

def foo(x,y): return x + y f = FPO.unapply(fn=foo, props=['x','y']) assert f(1,2) == 3

</section>

<details class="source"><summary>Source code</summary>

    def unapply(fn, props):
        '''
        ## FPO.unapply(..)
        Wraps a function to gather individual positional arguments into an object argument.
        ### Arguments:
            fn:     function to wrap
            props:  list of property names (strings) to indicate the order to gather individual positional arguments as properties.
        ### Returns:
            function
        Example:
            def foo(x,y):
                return x + y
            f = FPO.unapply(fn=foo, props=['x','y'])
            assert f(1,2) == 3
        '''
        def unapplied(*args):
            g = zip(props,args)
            kwargs = dict(g)
            return fn(**kwargs)
        return unapplied

</details></dd>

<dt id="fpo.unary">`<span>def <span class="ident">unary</span></span>(<span>fn, prop)</span>`</dt>

<dd>

<section class="desc">

## FPO.unary(..)

Wraps a function to restrict its inputs to only one named argument as specified.

### Arguments:

    fn: function to wrap
    prop: property name to allow as named argument

### Returns:

    function

### Example:

    def foo(**kwargs):
        return kwargs
    f = FPO.unary(fn=foo, prop='y')
    assert f(x=1,y=2,z=3) == {'y':2}

</section>

<details class="source"><summary>Source code</summary>

    def unary(fn,prop):
        '''
        ## FPO.unary(..)
        Wraps a function to restrict its inputs to only one named argument as specified.
        ### Arguments:
            fn: function to wrap
            prop: property name to allow as named argument
        ### Returns:
            function
        ### Example:
            def foo(**kwargs):
                return kwargs
            f = FPO.unary(fn=foo, prop='y')
            assert f(x=1,y=2,z=3) == {'y':2}
        '''
        def unary_fn(**kwargs):
            l_kwargs = {}
            l_kwargs[prop] = kwargs[prop]
            return fn(**l_kwargs)
        return unary_fn

</details></dd>

<dt id="fpo.uncurry">`<span>def <span class="ident">uncurry</span></span>(<span>fn)</span>`</dt>

<dd>

<section class="desc">

## FPO.uncurry(…)

Wraps a (strictly) curried function in a new function that accepts all the arguments at once, and provides them one at a time to the underlying curried function.

### Arguments:

    fn: function to uncurry

### Returns:

    function

### Example:

    def foo(x,y,z):
        return x + y + z
    f = FPO.curry(fn=foo, n=3)
    p = FPO.uncurry(fn=f)
    assert p(x=1,y=2,z=3) == 6

</section>

<details class="source"><summary>Source code</summary>

    def uncurry(fn):
        '''
        ## FPO.uncurry(...)
        Wraps a (strictly) curried function in a new function that accepts all the arguments at once, and provides them one at a time to the underlying curried function.
        ### Arguments:
            fn: function to uncurry
        ### Returns:
            function
        ### Example:
            def foo(x,y,z):
                return x + y + z
            f = FPO.curry(fn=foo, n=3)
            p = FPO.uncurry(fn=f)
            assert p(x=1,y=2,z=3) == 6
        '''
        def uncurry_fn(**kwargs):
            print('AAAA', kwargs)
            r = fn
            for key,v in kwargs.items():
                r = r(**{key:v})
            return r
        return uncurry_fn

</details></dd>

</dl>

</section>

</article>

<nav id="sidebar">

# Index

*   ### [Functions](#header-functions)

    *   `[ap](#fpo.ap "fpo.ap")`
    *   `[apply](#fpo.apply "fpo.apply")`
    *   `[binary](#fpo.binary "fpo.binary")`
    *   `[chain](#fpo.chain "fpo.chain")`
    *   `[chain_dict](#fpo.chain_dict "fpo.chain_dict")`
    *   `[complement](#fpo.complement "fpo.complement")`
    *   `[compose](#fpo.compose "fpo.compose")`
    *   `[constant](#fpo.constant "fpo.constant")`
    *   `[curry](#fpo.curry "fpo.curry")`
    *   `[curry_multiple](#fpo.curry_multiple "fpo.curry_multiple")`
    *   `[filter_in](#fpo.filter_in "fpo.filter_in")`
    *   `[filter_in_dict](#fpo.filter_in_dict "fpo.filter_in_dict")`
    *   `[filter_out](#fpo.filter_out "fpo.filter_out")`
    *   `[filter_out_dict](#fpo.filter_out_dict "fpo.filter_out_dict")`
    *   `[flat_map](#fpo.flat_map "fpo.flat_map")`
    *   `[flat_map_dict](#fpo.flat_map_dict "fpo.flat_map_dict")`
    *   `[flatten](#fpo.flatten "fpo.flatten")`
    *   `[head](#fpo.head "fpo.head")`
    *   `[identity](#fpo.identity "fpo.identity")`
    *   `[keep](#fpo.keep "fpo.keep")`
    *   `[keep_dict](#fpo.keep_dict "fpo.keep_dict")`
    *   `[map_dict](#fpo.map_dict "fpo.map_dict")`
    *   `[map_list](#fpo.map_list "fpo.map_list")`
    *   `[memoise](#fpo.memoise "fpo.memoise")`
    *   `[n_ary](#fpo.n_ary "fpo.n_ary")`
    *   `[partial](#fpo.partial "fpo.partial")`
    *   `[pick](#fpo.pick "fpo.pick")`
    *   `[pick_all](#fpo.pick_all "fpo.pick_all")`
    *   `[pipe](#fpo.pipe "fpo.pipe")`
    *   `[pluck](#fpo.pluck "fpo.pluck")`
    *   `[prop](#fpo.prop "fpo.prop")`
    *   `[reassoc](#fpo.reassoc "fpo.reassoc")`
    *   `[reduce](#fpo.reduce "fpo.reduce")`
    *   `[reduce_dict](#fpo.reduce_dict "fpo.reduce_dict")`
    *   `[reduce_right](#fpo.reduce_right "fpo.reduce_right")`
    *   `[reject](#fpo.reject "fpo.reject")`
    *   `[remap](#fpo.remap "fpo.remap")`
    *   `[set_prop](#fpo.set_prop "fpo.set_prop")`
    *   `[tail](#fpo.tail "fpo.tail")`
    *   `[take](#fpo.take "fpo.take")`
    *   `[trampoline](#fpo.trampoline "fpo.trampoline")`
    *   `[transduce_fn](#fpo.transduce_fn "fpo.transduce_fn")`
    *   `[transducer_bool_and](#fpo.transducer_bool_and "fpo.transducer_bool_and")`
    *   `[transducer_bool_or](#fpo.transducer_bool_or "fpo.transducer_bool_or")`
    *   `[transducer_default](#fpo.transducer_default "fpo.transducer_default")`
    *   `[transducer_filter](#fpo.transducer_filter "fpo.transducer_filter")`
    *   `[transducer_filter_fn](#fpo.transducer_filter_fn "fpo.transducer_filter_fn")`
    *   `[transducer_into](#fpo.transducer_into "fpo.transducer_into")`
    *   `[transducer_into_fn](#fpo.transducer_into_fn "fpo.transducer_into_fn")`
    *   `[transducer_list](#fpo.transducer_list "fpo.transducer_list")`
    *   `[transducer_map](#fpo.transducer_map "fpo.transducer_map")`
    *   `[transducer_map_fn](#fpo.transducer_map_fn "fpo.transducer_map_fn")`
    *   `[transducer_number](#fpo.transducer_number "fpo.transducer_number")`
    *   `[transducer_string](#fpo.transducer_string "fpo.transducer_string")`
    *   `[transducer_transduce](#fpo.transducer_transduce "fpo.transducer_transduce")`
    *   `[unapply](#fpo.unapply "fpo.unapply")`
    *   `[unary](#fpo.unary "fpo.unary")`
    *   `[uncurry](#fpo.uncurry "fpo.uncurry")`

</nav>

</main>

<footer id="footer">

Generated by [<cite>pdoc</cite> 0.5.3](https://pdoc3.github.io/pdoc).

</footer>

<script>hljs.initHighlightingOnLoad()</script>


## References
https://github.com/mitmproxy/pdoc
https://docs.python.org/3.7/library/index.html
https://pypi.org/
http://flask.pocoo.org/docs/1.0/tutorial/layout/
https://docs.python.org/3.6/howto/functional.html
https://www.oreilly.com/ideas/2-great-benefits-of-python-generators-and-how-they-changed-me-forever
https://stackoverflow.com/questions/102535/what-can-you-use-python-generator-functions-for
https://data-flair.training/blogs/advantages-and-disadvantages-of-python/
https://github.com/getify/FPO/blob/master/docs/core-API.md#fpocomplement
