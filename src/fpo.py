
'''
##FPO.ap(...)
Produces a new list that is a concatenation of sub-lists, each produced by calling FPO.map(..) with each mapper function and the main list.

###Args:
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

###Args:
    fn: function to wrap
    props: (optional) list of property names (strings) to indicate the order to spread properties as individual arguments. If omitted, the signature of fn is parsed for its parameter list to try to determine an ordered property list; this detection only works for simple parameters (including those with default parameter value settings).


###Returns:
    function

###Example:
    def foo(x,y = 2): return x + y
    def bar([a,b],c): return a + b + c

    f = FPO.apply(fn: foo)
    p = FPO.apply(fn: bar, props:["x","y"]);

    f({x: 1, y:1})   // 2
    f({x: 3})         // 5
    p({x:[1,2], y: 3}))    // 6
'''
def apply(fn, props):
    pass



'''
##FPO.pluck(...)
Plucks properties form the given list and return a list of properties' values

###Args:
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

