"""TSTE: t-Distributed Stochastic Triplet Embedding

X = tste(triplets, no_dims=2, lambda=0, alpha=no_dims-1, use_log=True, verbose=True)

where 'triplets' is an integer array with shape (k, 3) containing the
known triplet relationships between points. Each point is refered to
by its index: consider a set of points and some (possibly nonmetric)
distance function d.

For each t in triplets' rows,
   assume d(point[t[0]], point[t[1]])
        < d(point[t[0]], point[t[2]])

The function implements t-distributed stochastic triplet embedding
(t-STE) based on the specified triplets, to construct an embedding
with no_dims dimensions. The parameter lambda specifies the amount of
L2- regularization (default = 0), whereas alpha sets the number of
degrees of freedom of the Student-t distribution (default = 1). The
variable use_log determines whether the sum of the log-probabilities
or the sum of the probabilities is maximized (default = true).

Note: This function directly learns the embedding.


Original MATLAB implementation: (C) Laurens van der Maaten, 2012, Delft University of Technology

Python and Theano port: (C) Michael Wilber, 2013, UC San Diego

"""
import numpy as np

from _tste import tste_grad_cython_log

def tste(triplets, no_dims=2, lamb=0, alpha=None, use_log=True,verbose=True, max_iter=1000, save_each_iteration=False, initial_X=None,static_points=np.array([]), normalize_gradient=False, ignore_zeroindexed_error=True):
    """Learn the triplet embedding for the given triplets.

    Returns an array with shape (max(triplets)+1, no_dims). The i-th
    row in this array corresponds to the no_dims-dimensional
    coordinate of the point.

    """
    if alpha is None:
        alpha = no_dims-1

    N = np.max(triplets) + 1
    assert -1 not in triplets

    # A warning to Matlab users:
    if not ignore_zeroindexed_error:
        assert 0 in triplets, "Triplets should be 0-indexed, not 1-indexed!"
    # Technically, this is allowed I guessss... if your triplets don't
    # refer to some points you need... Just don't say I didn't warn
    # you. Remove this assertion at your own peril!

    n_triplets = len(triplets)

    # Initialize some variables
    if initial_X is None:
        X = np.random.randn(N, no_dims).astype('float32') * 0.0001
    else:
        X = initial_X

    C = np.Inf
    tol = 1e-7              # convergence tolerance
    eta = 2.                # learning rate
    best_C = np.Inf         # best error obtained so far
    best_X = X              # best embedding found so far
    iteration_Xs = []       # for debugging ;) *shhhh*


    # Perform main iterations
    iter = 0; no_incr = 0;
    while iter < max_iter and no_incr < 5:
        old_C = C;
        # Calculate gradient descent and cost
        assert use_log
        C,G = tste_grad_cython_log(X, N, no_dims, triplets, lamb, alpha)

        if C < best_C:
            best_C = C
            best_X = X

        # Perform gradient update
        if save_each_iteration:
            iteration_Xs.append(X.copy())

        # (NEW:) Optionally normalize each point's gradient by the
        # number of times that the point appears in the triplets
        if normalize_gradient:
            prior = np.bincount(triplets.ravel(),
                                minlength=N)
            prior[prior==0] = 1
            # if prior[i]==0, then the point has no gradient anyway
            G = G / (prior/np.linalg.norm(prior))[:,np.newaxis]
        X = X - (float(eta) / n_triplets * N) * G
        if len(static_points):
            X[static_points] = initial_X[static_points]

        # Update learning rate
        if old_C > C + tol:
            no_incr = 0
            eta *= 1.01
        else:
            no_incr = no_incr + 1
            eta *= 0.5

        # Print out progress
        iter += 1
        if verbose and iter%10 == 0:
            # These are Euclidean distances:
            sum_X = np.sum(X**2, axis=1)
            D = -2 * (X.dot(X.T)) + sum_X[np.newaxis,:] + sum_X[:,np.newaxis]
            # ^ D = squared Euclidean distance?
            no_viol = np.sum(D[triplets[:,0],triplets[:,1]] > D[triplets[:,0],triplets[:,2]]);
            print "Iteration ",iter, ' error is ',C,', number of constraints: ', (float(no_viol) / n_triplets)

    if save_each_iteration:
        return iteration_Xs
    return best_X
