    
'''
##FPO.pluck(...)
Plucks properties form the given list and return a list of properties' values

###Args:
    list:   list
    *args:  properties

###Returns:    
    a list of value

###Example:
    list = [{'x': 1, 'y':2}, {'x': 3, 'y': 4}]
    result = pluck(d, 'x') #[1,3]
'''
# pluck = lambda d, *args: [d[arg] for arg in args]
def pluck(list, *args):
    fn = lambda d, *args: [d[arg] for arg in args]
    r = [fn(o, *args) for o in list]
    if len(args) == 1:
        return [v[0] for v in r]
    else:
        return r

