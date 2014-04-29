import numpy as np
from numpy.linalg import inv
from scipy import sparse, optimize


def f(V, *args):
    shape, R, W, U = args
    V = np.matrix(np.reshape(V, shape))
    Res = np.multiply(W, R - np.dot(U, V.T) )
    res = np.ravel(Res)
    return np.dot(res,res)

def gradf(V, *args):
    shape, R, W, U = args
    V = np.matrix(np.reshape(V, shape))
    Res = 2 * np.multiply(W.T, V*U.T - R.T) * U
    res = np.ravel(Res)
    return np.dot(res,res)

def error2(R, W, U, V):
    Res = np.multiply(W, R - np.dot(U, V.T) )
    res = np.ravel(Res)
    return np.dot(res,res)    

def line_search(res, R, W, U, V):
    decay = 0.1
    Res = np.ravel(res)
    err0 = np.dot(Res,Res)
    alpha0 = 0.00001
    for i in range(50):
        alpha = decay * 1.0 / (i + 1)**2
        err = error2(R, W, U, V + 2 * alpha * res * U)
        if err < err0:
            err0 = err
            alpha0 = alpha
    return alpha0

def weighted_low_rank_factorization(R, K, W=None, steps=1000, alpha=0.001, tol=0.01):
    ### type check for matrices?? ###

    # if weights not given set to 0/1 
    if not W:
        W = np.array(R, copy=True)
        W[ W > 0 ] = 1

    if R.shape != W.shape:
        print "ERROR: Matrices shapes do not match"
        return -1

    # improve initialization, smaller numbers, avg, or rand svd?
    U = np.matrix(np.random.rand(R.shape[0],K)) 
    V = np.matrix(np.random.rand(R.shape[1],K)) 
    avg = (R.sum(0) * 1.0)/(R!=0).sum(0)
    V[:,0] = avg.T

    iters = 0 
    converged = False

    res = np.multiply(W, R - U*V.T)
    err = np.ravel(res)
    sse = np.dot(err, err)
    print "%d\t%f" %(iters, sse)

    while iters < steps and not converged:
        # compute minimal U in terms of fixed V
        for i in xrange(len(U)):
            w = np.diag(W[i])
            U[i] = (inv(V.T * w * V) * V.T * w * R[i].T ).T

        for i in xrange(len(V)):
            w = np.diag(W[:,i])
            V[i] = (inv(U.T * w * U) * U.T * w * R[:,i] ).T

#        shape = V.shape
#        V = optimize.fmin_cg(f, V.ravel(), fprime=gradf, args=(shape, R, W, U))
#        V = np.matrix(np.reshape(V, shape))
#        for i in xrange(10):
#            res = np.multiply(W.T, V*U.T - R.T)
#            alpha = line_search(res, R, W, U, V)
#            print alpha
#            V = V + 2 * alpha * res * U

        res = np.multiply(W, R - U*V.T)
        err = np.ravel(res)
        sse = np.dot(err, err)
        print "%d\t%f" %(iters, sse)
        
        if sse < tol:
            converged = True
        iters += 1

    return U, V


#def als_factorization(R, K,

# working for dense
def d_factorize(R, K, steps=1000, alpha=0.0001):
    iters = 0
    U = np.random.rand(R.shape[0],K)
    V = np.random.rand(R.shape[1],K)
    W = np.array(R, copy=True)
    W[ W > 0 ] = 1 
    while iters < steps:
#        Res = np.array(W) * np.array(R - np.dot(U, V.T))
        Res = np.multiply(W, R - np.dot(U, V.T) )
        res = np.ravel(Res)
        SqErr = np.dot(res,res)
        if iters % 1 == 0:
            print "%d\t%f" %(iters, SqErr)
        if SqErr < 0.001:
            break

        for i in xrange(len(U)):
            U[i] = U[i] +  alpha * np.dot(Res[i], V)
#        Res = np.array(W) * np.array(R - np.dot(U, V.T))
        Res = np.multiply(W, R - np.dot(U, V.T) )
        for i in xrange(len(V)):
            V[i] = V[i] +  alpha * np.dot(Res[:,i], U)

        iters += 1

    return U, V

"""
def weighted_svd(X, L, K, steps=100):
    iters = 0
    W = np.array(X, copy=True)
    W[ W > 0 ] = 1 
    R = np.zeros(X.shape)
    while iters < steps:
        continue
"""

def s_factorize(X, I, J, K, steps=1000, alpha=0.001):
    iters = 0
    U = np.random.rand(X.shape[0], K)
    V = np.random.rand(X.shape[1], K)
    Res = sparse.coo_matrix((np.ones(len(I)),(I,J)),shape=X.shape).tocsr()
    tmp = np.zeros(len(I))

    # intialize gradient
    for indx, i, j in zip(xrange(len(I)), I, J):
        tmp[indx] = U[i].dot(V.T[:,j])
    Res.data = X.data - tmp
    SqErr = sum( (Res.data)**2 ) 

    while iters < steps:

        if iters % 50 == 0:
            print "%d\t%.10f\t%f" %(iters, alpha, SqErr)
        if SqErr < 0.001:
            break
        prev_SqErr = SqErr

        # update U
        for i in xrange(len(U)):
            U[i] = U[i] + 2 * alpha * np.dot(Res[i].todense(), V)

        # recalculate gradient
        for indx, i, j in zip(xrange(len(I)), I, J):
            tmp[indx] = U[i].dot(V.T[:,j])
        Res.data = X.data - tmp
        SqErr = sum( (Res.data)**2 ) 

        if SqErr > prev_SqErr:
            alpha /= 2.0
        prev_SqErr = SqErr

        # update V
        for i in xrange(len(V)):
            V[i] = V[i] + 2 * alpha * np.dot(Res[:,i].todense().T, U)

        # recalculate gradient
        for indx, i, j in zip(xrange(len(I)), I, J):
            tmp[indx] = U[i].dot(V.T[:,j])
        Res.data = X.data - tmp
        SqErr = sum( (Res.data)**2 ) 

        if SqErr > prev_SqErr:
            alpha /= 2.0
        if prev_SqErr - SqErr < .01:
            alpha *= 2.0

        if alpha < 0.0000001:
            alpha = 0.0000001
        
        iters += 1

    return U, V


X = np.matrix(np.random.random_integers(0,20,size=(500,600)))
X[X > 5] = 0
weighted_low_rank_factorization(X, 8, steps=100)
