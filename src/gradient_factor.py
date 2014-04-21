import numpy as np
from scipy import sparse
"""
Gradient Descent for Weighted Matrix Factorization

X = np.array([ [5,3,0,1],
               [4,0,0,1],
               [1,1,0,5],
               [0,1,5,4]]
              )
W = np.array([ [1,1,0,1],
               [1,0,0,1],
               [1,1,0,1],
               [0,1,1,1]]
            )
U = np.random.rand(4,2)
V = np.random.rand(4,2)

u, v = factorize(X,U,V,W)
np.dot(u, v.T) 

"""

# working for dense
def d_factorize(X, K, steps=1000, alpha=0.0001):
    iters = 0
    U = np.random.rand(X.shape[0],K)
    V = np.random.rand(X.shape[1],K)
    W = np.array(X, copy=True)
    W[ W > 0 ] = 1 
    while iters < steps:
        Res = np.array(W) * np.array(X - np.dot(U, V.T))
        SqErr = sum(sum( Res**2 )) 
        if iters % 1 == 0:
            print "%d\t%f" %(iters, SqErr)
        if SqErr < 0.001:
            break

        for i in xrange(len(U)):
            U[i] = U[i] + 2 * alpha * np.dot(Res[i], V)
        Res = np.array(W) * np.array(X - np.dot(U, V.T))
        for i in xrange(len(V)):
            V[i] = V[i] + 2 * alpha * np.dot(Res[:,i], U)
        
        iters += 1

    return U, V


def f(Data, I, J, A, B):
    return sum(( Data - [ A[i].dot(B.T[:,j]) for i, j in zip(I, J) ] )**2)

def fprime(res, A, B):
    for i in xrange(len(A)):
        A[i] = B[i] + 2

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


