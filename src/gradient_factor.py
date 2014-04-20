import numpy as np

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


def factorize(X, U, V, W, steps=1000, alpha=0.001):
    iters = 0
    converged = False
    while iters < steps:
        Res = W * (X - np.dot(U, V.T))
        SqErr = sum(sum( Res**2 )) 
        if iters % 50 == 0:
            print "%d\t%f" %(iters, SqErr)
        if SqErr < 0.001:
            break

        for i in xrange(len(U)):
            U[i] = U[i] + 2 * alpha * np.dot(Res[i], V)
        Res = W * (X - np.dot(U, V.T))
        for i in xrange(len(V)):
            V[i] = V[i] + 2 * alpha * np.dot(Res[:,i], U)
        
        iters += 1

    return U, V
