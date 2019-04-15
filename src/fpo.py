    
'''
##FPO.pluck(...)
Plucks properties form the given dictionary and return a new dictionary

###Args:
    d:      dictionary
    *args:  properties

###Returns:    
    a new dictionary

###Example:
    d = {'x':1, 'y':2, 'z': 3}
    result = pluck(d, 'x', 'y') #{'x':1, 'y':2}
'''
pluck = lambda d, *args: (d[arg] for arg in args)

