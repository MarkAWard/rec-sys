import numpy as np

def truncate(R):
    R2 = np.copy(R)
    R2[ R > 5 ] = 5
    R2[ R < 1 ] = 1
    return R2

def linear(R, low=1.0, high=5.0):
    mx = R.max()
    mn = R.min()
    m = (high - low)/(mx - mn)
    b = high - m * mx
    return (m*R + b)

def buckets(R, one=5555, two=10788, three=22429, four=45622, five=30142):
    d = {0:one, 1:two, 2:three, 3:four, 4:five}
    indx = [0]
    total = float(one + two + three + four + five)
    
    try:
        R = R.todense()
        vals = R.ravel()
    except:
        vals = R.ravel()
    vals = np.sort(vals)
    num = max(vals.shape)
    vals = vals.reshape(num,1)

    for i in range(5):
        indx.append( indx[i] + int( num * (d[i] / total)) )
    if indx[5] < num:
        indx[5] = num - 1

    tmp = np.copy(R)
    for i in range(5):
        tmp[ np.logical_and( R >= vals[indx[i]], R <= vals[indx[i+1]] ) ] = i +1
        
    return tmp
