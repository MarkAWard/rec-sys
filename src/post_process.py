import numpy as np
import matplotlib.pyplot as plt

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

#def intervals(R):
# round to integers or .5 intervals or .25 intervals...

def buckets(R, one=5555, two=10788, three=22429, four=45622, five=30142):
    d = {0:one, 1:two, 2:three, 3:four, 4:five}
    indx = [0]
    total = float(one + two + three + four + five)
    
    try:
        vals = R.ravel()
    except:
        R = R.todense()
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

def calibrate(R, cv, knn=15, bins=500):
    
    points = create_calibration_points(R, cv, bins=bins)
    
    for i in range(0, R.shape[0]):
        for j in range(0, R.shape[1]):
            indx = find_point(R[i,j], points)
            R[i,j] = 


def create_calibration_points(R, cv, bins=500):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    mn = R.min()
    mx = R.max()

    for indx, val in enumerate(cv.data):
        i = cv.nonzero()[0][indx]
        j = cv.nonzero()[1][indx]
        y1.append(val)
        pred = R[i,j]
        x1.append(pred)
        
        for k in np.linspace(mn-.01, mx+.01, num=bins):
            if pred <= k:
                x2.append(k)
                break

    y1 = np.array(y1)
    y2 = np.copy(y1)
    for k in np.linspace(mn-.01, mx+.01, num=bins):
        mask = np.array(x2)==k
        y2[mask] = np.average(y1[mask])

    z = [(i, j) for i, j in zip(x2, y2) if j > 0]
    
    return np.unique(z)
    
def find_point(pred, points):
    indx = 0
    while indx < len(points) and points[indx,0] < pred:
        indx += 1
    return indx
    

def calibrate_plot(R, cv, bins=500):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    mn = R.min()
    mx = R.max()

    for indx, val in enumerate(cv.data):
        i = cv.nonzero()[0][indx]
        j = cv.nonzero()[1][indx]
        y1.append(val)
        pred = R[i,j]
        x1.append(pred)
        
        for k in np.linspace(mn-.01, mx+.01, num=bins):
            if pred <= k:
                x2.append(k)
                break

    y1 = np.array(y1)
    y2 = np.copy(y1)
    for k in np.linspace(mn-.01, mx+.01, num=bins):
        mask = np.array(x2)==k
        y2[mask] = np.average(y1[mask])

    plt.scatter(x1,y1)
    plt.show()
    plt.scatter(x2,y2)
    plt.xlabel("Predicted Rating")
    plt.ylabel("Average True Rating")
    plt.title("Calibration Plot")
    plt.show()
