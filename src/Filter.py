def filter_exp(expr, obj):
    expr = expr.replace('(', ' ( ').replace(')',' ) ')
    words = re.split(r'\W+', expr)
    values= {}
    # create a dict for desired keys from obj
    for word in words:
        try:
            values[word] = obj[word]
        except KeyError:
            # ignore key errors here
            continue
    return parse_expression(expr, values)

def parse_expression(expr, values):
    """
    
    """
    # copy expr to create final boolean conditional
    result = expr
    close = expr.find(')')
    ind1 = -1
    ind2 = 1
    # find each set ( l_operand op r_operand )
    while close > 0 and ind2 > 0:
        ind1 = expr.find('(', ind1+1 )
        ind2 = expr.find('(', ind1+1 )
        while ind2 > 0 and ind2 < close:
            ind1 = ind2
            ind2 = expr.find('(', ind1+1 )
        # found first set of matching parens
        term = expr[ind1+1:close].split()
        # pass each term and operator to be evaluated
        res = evaluate( values[term[0]], term[1], values[term[2]] )
        # res = True/False, replace term in result with res 
        result = result.replace(expr[ind1+1:close], str(res))
        # find next closing paren and iterate
        close = expr.find(')', close+1)
    # result looks like: ( ( True ) and ( ( True ) or ( False )  )
    return eval(result)

def evaluate(left, op, right):
    switch = {
        '<': lambda left, right: left < right, 
        '<=': lambda left, right: left <= right, 
        '>': lambda left, right: left > right,
        '>=': lambda left, right: left >= right,
        '==': lambda left, right: left == right, 
        '!=': lambda left, right: left != right, 
        'in': lambda left, right: left in right, 
        'and': lambda left, right: left and right, 
        'or': lambda left, right: left or right 
        }
    return switch[op](left, right)
