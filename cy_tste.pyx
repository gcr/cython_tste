#cython: boundscheck=False, wraparound=False, cdivision=True

"""TSTE: t-Distributed Stochastic Triplet Embedding

X = tste(triplets, no_dims=2, lambda=0, alpha=no_dims-1, verbose=True)

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
degrees of freedom of the Student-t distribution (default = 1).

Note: This function directly learns the embedding.


Original MATLAB implementation: (C) Laurens van der Maaten, 2012, Delft University of Technology

Ports to Python, Theano, and Cython: (C) Michael Wilber, 2013-2014, UCSD and Cornell

"""

cimport numpy as cnp
import numpy as np
from libc.math cimport log
cimport cython.parallel
cimport openmp

cpdef tste_grad(npX,
                int N,
                int no_dims,
                long [:, ::1] triplets,
                double lamb,
                double alpha,
                int use_log,

                # These are just here to avoid allocation. :/
                double[::1] sum_X, # = np.zeros((N,), dtype='float64')
                double[:, ::1] K, # = np.zeros((N, N), dtype='float64')
                double[:, ::1] Q, # = np.zeros((N, N), dtype='float64')
                double [:, ::1] dC, # = np.zeros((N, no_dims), 'float64')
                double[:, :, ::1] dC_part, #= np.zeros((no_triplets, no_dims, 3), 'float64')

):
    """ Compute the cost function and gradient update of t-STE """

    cdef long[:] triplets_A = triplets[:,0]
    cdef long[:] triplets_B = triplets[:,1]
    cdef long[:] triplets_C = triplets[:,2]
    cdef int i,t,j,k
    cdef double[:, ::1] X = npX
    cdef unsigned int no_triplets = len(triplets)
    cdef double P = 0
    cdef double C = 0
    cdef double A_to_B, A_to_C, const

    # L2 Regularization cost
    C += lamb * np.sum(npX**2)

    # Compute student-T kernel for each point
    # i,j range over points; k ranges over dims
    with nogil:
        for i in xrange(N):
            sum_X[i] = 0
            for k in xrange(no_dims):
                # Squared norm
                sum_X[i] += X[i,k]*X[i,k]
        for i in cython.parallel.prange(N):
            for j in xrange(N):
                K[i,j] = sum_X[i] + sum_X[j]
                for k in xrange(no_dims):
                    K[i,j] += -2 * X[i,k]*X[j,k]
                Q[i,j] = (1 + K[i,j] / alpha) ** -1
                K[i,j] = (1 + K[i,j] / alpha) ** ((alpha+1)/-2)
                # Now, K[i,j] = ((sqdist(i,j)/alpha + 1)) ** (-0.5*(alpha+1)),
                # which is exactly the numerator of p_{i,j} in the lower right of
                # t-STE paper page 3.
                # The proof follows because sqdist(a,b) = (a-b)(a-b) = a^2+b^2-2ab

        # Compute probability (or log-prob) for each triplet
        for t in cython.parallel.prange(no_triplets):
            P = K[triplets_A[t], triplets_B[t]] / (
                K[triplets_A[t],triplets_B[t]] +
                K[triplets_A[t],triplets_C[t]])
            # This is exactly p_{ijk}, which is the equation in the lower-right
            # of page 3 of the t-STE paper.
            C += -log(P) if use_log else -P
            # This is exactly the cost.

            for i in xrange(no_dims):
                # For i = each dimension to use
                # Calculate the gradient of *this triplet* on its points.
                const = (alpha+1) / alpha
                A_to_B = ((1 - P) *
                          # K[triplets_A[t],triplets_B[t]] *
                          Q[triplets_A[t],triplets_B[t]] *
                          (X[triplets_A[t], i] - X[triplets_B[t], i]))
                A_to_C = ((1 - P) *
                          # (K[triplets_A[t],triplets_C[t]]) *
                          Q[triplets_A[t],triplets_C[t]] *
                          (X[triplets_A[t], i] - X[triplets_C[t], i]))

                # Problem: Since this is a parallel for loop, we can't
                # accumulate everything at once. Race conditions.
                # So I calculate it once here:
                if use_log:
                    dC_part[t, i, 0] = -const * (A_to_B - A_to_C)
                    dC_part[t, i, 1] = -const * (-A_to_B)
                    dC_part[t, i, 2] = -const * (A_to_C)
                else:
                    dC_part[t, i, 0] = -const * P * (A_to_B - A_to_C)
                    dC_part[t, i, 1] = -const * P * (-A_to_B)
                    dC_part[t, i, 2] = -const * P * (A_to_C)

        # ...and then accumulate:
        for n in xrange(N):
            for i in xrange(no_dims):
                dC[n, i] = 0
        for t in xrange(no_triplets):
            for i in xrange(no_dims):
                dC[triplets_A[t], i] += dC_part[t, i, 0]
                dC[triplets_B[t], i] += dC_part[t, i, 1]
                dC[triplets_C[t], i] += dC_part[t, i, 2]
        for n in xrange(N):
            for i in xrange(no_dims):
                # The 2*lamb*npx is for regularization: derivative of L2 norm
                dC[n,i] = (dC[n,i]*-1) + 2*lamb*X[n,i]
    return C

def tste(triplets,
         no_dims=2,
         lamb=0,
         alpha=None,
         verbose=True,
         max_iter=1000,
         save_each_iteration=False,
         initial_X=None,
         static_points=np.array([]),
         ignore_zeroindexed_error=True,
         num_threads=None,
         use_log=False,
):
    """Learn the triplet embedding for the given triplets.

    Returns an array with shape (max(triplets)+1, no_dims). The i-th
    row in this array corresponds to the no_dims-dimensional
    coordinate of the point.

    Parameters:

    triplets: An Nx3 integer array of object indices. Each row is a
              triplet; first column is the 'reference', second column
              is the 'near edge', and third column is the 'far edge'.
    use_log:  When set, optimize for \sum\log p; when unset, optimize
              just \sum p. This changes the algorithm results quite a
              bit, so try both cases and pick which works better. Note
              that tSTE is a special case of Crowd Kernel Learning by
              construction when use_log=True, no_dims=2, alpha=1.
    no_dims:  Number of dimensions in final embedding. High-dimensional
              embeddings are much easier to satisfy (lower training
              error), but may capture less information.
    lamb:     L2 regularization constant. Collapses points closer to origin.
    alpha:    Degrees of freedom in student T kernel. Default is no_dims-1.
              Considered "black magic"; roughly, how much of an impact
              badly satisfying points have on the gradient calculation.
    verbose:  Prints log messages every 10 iterations
    save_each_iteration: When true, will save intermediate results to
              a list so you can watch it converge.
    initial_X: The initial set of points to use. Normally distributed if unset.
    static_points: List of static points that will have their gradient zeroed.
    ignore_zeroindexed_error: For the MATLAB-weary
    num_threads: Parallelism.

    """
    if num_threads is None:
        num_threads = openmp.omp_get_num_procs()
    openmp.omp_set_num_threads(num_threads)

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
        X = np.random.randn(N, no_dims) * 0.0001
    else:
        X = initial_X

    C = np.Inf
    tol = 1e-7              # convergence tolerance
    eta = 2.                # learning rate
    best_C = np.Inf         # best error obtained so far
    best_X = X              # best embedding found so far
    iteration_Xs = []       # for debugging ;) *shhhh*
    G = np.zeros((N, no_dims), 'float64')

    # Avoid allocation at readability cost. Bah. I don't like it but
    # it *does* avoid MemoryErrors on my machine, so a net win I guess.
    _sum_X = np.zeros((N,), dtype='float64')
    _K = np.zeros((N, N), dtype='float64')
    _Q = np.zeros((N, N), dtype='float64')
    _dC_part= np.zeros((len(triplets), no_dims, 3), 'float64')

    # Perform main iterations
    iter = 0; no_incr = 0;
    while iter < max_iter and no_incr < 5:
        old_C = C
        # Calculate gradient descent and cost
        C = tste_grad(X, N, no_dims, triplets, lamb, alpha, use_log,
                        _sum_X, _K, _Q, G, _dC_part
        )

        X = X - (float(eta) / n_triplets * N) * G
        if len(static_points):
            X[static_points] = initial_X[static_points]

        if C < best_C:
            best_C = C
            best_X = X

        # Perform gradient update
        if save_each_iteration:
            iteration_Xs.append(X.copy())

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
