import numpy as np
from numpy.linalg import inv
from scipy import sparse, optimize

def weighted_low_rank_factorization(R, K, W=None, steps=1000, method='als', lambd=0.1, alpha=.0001, tol=0.001):
    """
    Compute the weighted rank-K factorization of a matrix R. Weights are given by matrix W
    If W is not provided it construsted as W_ij = 1 where R_ij > 0 else W_ij = 0
    Computed using alternating least squares us 'steps' number of iterations or until
    the desired tolerance is reached (if it is obtained before 'steps' iterations)
    Uses Tikhonov regularization with parameter 'lambd' to penalize large parameters and
    also ensures the inverse is not (attempted to be) computed for a singular matrix due 
    to extreme sparsity.
    """ 

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
    N = sum(sum(W))

    iters = 0 
    converged = False

    res = np.multiply(W, R - U*V.T)
    err = np.ravel(res)
    sse = np.dot(err, err)
    print "%d\t%f" %(iters, sse/N)

    while iters < steps and not converged:

        # compute U
        if method == 'als' or method == 'mix':
            for i in xrange(len(U)):
                w = np.diag(W[i])
                U[i] = (inv(V.T * w * V + lambd * np.eye(K)) * V.T * w * R[i].T ).T
        
        if method == 'nmf':
            num = R * V
            denom = U * V.T * V
            U = np.multiply(U, np.divide(num, denom) )

        # compute V
        if method == 'als':
            for i in xrange(len(V)):
                w = np.diag(W[:,i])
                V[i] = (inv(U.T * w * U + lambd * np.eye(K)) * U.T * w * R[:,i] ).T
        
        if method == 'mix':
            for subiter in xrange(1000):
                res = (np.multiply(W.T, V*U.T - R.T)).T
                for i in xrange(len(V)):
                    V[i] = V[i] +  alpha * np.dot(res[:,i].T, U)
                    if subiter % 10 == 0:
                        res = np.multiply(W, R - U*V.T)
                        err = np.ravel(res)
                        sse = np.dot(err, err)
                        print "\t%d\t%f" %(subiter, sse/N)

        if method == 'nmf':
            num = U.T * R
            denom = U.T * U * V.T
            V = (np.multiply(V.T, np.divide(num, denom) )).T


        res = np.multiply(W, R - U*V.T)
        err = np.ravel(res)
        sse = np.dot(err, err)
        print "%d\t%f" %(iters, sse/N)
        
        if sse/N < tol:
            converged = True
        iters += 1

    return U, V


def SGD(Row, Col, Ratings, shape, K, steps=1000, alpha=0.001, lambd=0.01, tol=0.001):

    U = np.matrix(np.random.rand(shape[0],K)) 
    V = np.matrix(np.random.rand(shape[1],K)) 

    iters = 0
    converged = False
    while iters < steps and not converged:
        res = 0.0
        for i, j, rating in zip(Row, Col, Ratings):
            err = rating - np.dot(U[i], V[j])
            U[i] = U[i] - alpha * (lambd * U[i] + err * V[j])
            V[j] = V[j] - alpha * (lambd * V[j] + err * U[i])
            
            res += np.abs( rating - np.dot(U[i], V[j]) )
        
        avg_sq_er = res/len(Ratings)
        if avg_sq_er < tol:
            converged = True
        print "%d\t%f" %(iters, avg_sq_er)

    return U, V


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


X = np.matrix(np.random.random_integers(0,200,size=(800,500)))
X[X > 5] = 0
a, b = weighted_low_rank_factorization(X, 10, method='nmf', steps=100, lambd=0.01, alpha=.0000001)
print X
print a * b.T
