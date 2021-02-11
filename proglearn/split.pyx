#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False

cimport cython

import numpy as np

from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map as ordered_map
from cython.operator import dereference, postincrement

from cython.parallel import prange

# Computes the gini score for a split
# 0 < t < len(y)

cdef class BaseObliqueSplitter:

    cdef void argsort(self, double[:] y, int[:] idx) nogil:
        cdef int length = y.shape[0]
        cdef int i = 0
        cdef ordered_map[double, int] sort_map
        cdef ordered_map[double, int].iterator it = sort_map.begin()

        for i in range(length):
            sort_map[y[i]] = i

        it = sort_map.begin()
        i = 0
        while it != sort_map.end():
            idx[i] = dereference(it).second
            i = i + 1
            postincrement(it)

    cdef (int, int) argmin(self, double[:, :] A) nogil:
        cdef int N = A.shape[0]
        cdef int M = A.shape[1]
        cdef int i = 0
        cdef int j = 0
        cdef int min_i = 0
        cdef int min_j = 0
        cdef double minimum = A[0, 0]

        for i in range(N):
            for j in range(M):

                if A[i, j] < minimum:
                    minimum = A[i, j]
                    min_i = i
                    min_j = j

        return (min_i, min_j)

    cdef int argmax(self, double[:] A) nogil:
        cdef int N = A.shape[0]
        cdef int i = 0
        cdef int max_i = 0
        cdef double maximum = A[0]

        for i in range(N):
            if A[i] > maximum:
                maximum = A[i]
                max_i = i

        return max_i

    cdef double impurity(self, double[:] y) nogil:
        cdef int length = y.shape[0]
        cdef double dlength = y.shape[0]
        cdef double temp = 0
        cdef double gini = 1.0
        
        cdef unordered_map[double, double] counts
        cdef unordered_map[double, double].iterator it = counts.begin()

        # Count all unique elements
        for i in range(0, length):
            temp = y[i]
            counts[temp] += 1

        it = counts.begin()
        while it != counts.end():
            temp = dereference(it).second
            temp = temp / dlength
            temp = temp * temp
            gini -= temp

            postincrement(it)

        return gini

    cdef double score(self, double[:] y, int t) nogil:
        cdef double length = y.shape[0]
        cdef double left_gini = 1.0
        cdef double right_gini = 1.0
        cdef double gini = 0
    
        cdef double[:] left = y[:t]
        cdef double[:] right = y[t:]

        cdef double l_length = left.shape[0]
        cdef double r_length = right.shape[0]

        left_gini = self.impurity(left)
        right_gini = self.impurity(right)

        gini = (l_length / length) * left_gini + (r_length / length) * right_gini
        return gini

    # X = proj_X, y = y_sample
    def best_split(self, double[:, :] X, double[:] y, int[:] sample_inds):

        cdef int n_samples = X.shape[0]
        cdef int proj_dims = X.shape[1]
        cdef int i = 0
        cdef int j = 0
        cdef long temp_int = 0;
        cdef double node_impurity = 0;

        cdef int thresh_i = 0
        cdef int feature = 0
        cdef double best_gini = 0
        cdef double threshold = 0
        cdef double improvement = 0
        cdef double left_impurity = 0
        cdef double right_impurity = 0

        Q = np.zeros((n_samples, proj_dims), dtype=np.float64)
        cdef double[:, :] Q_view = Q

        idx = np.zeros(n_samples, dtype=np.intc)
        cdef int[:] idx_view = idx

        y_sort = np.zeros(n_samples, dtype=np.float64)
        cdef double[:] y_sort_view = y_sort
        
        feat_sort = np.zeros(n_samples, dtype=np.float64)
        cdef double[:] feat_sort_view = feat_sort

        feat_idx = np.zeros(n_samples, dtype=np.intc) 
        cdef int[:] feat_idx_view = feat_idx

        si_return = np.zeros(n_samples, dtype=np.intc)
        cdef int[:] si_return_view = si_return
        
        # No split = just impurity of the whole thing
        node_impurity = self.impurity(y)
        Q_view[0, :] = node_impurity
        Q_view[n_samples - 1, :] = node_impurity

        for j in range(0, proj_dims):
       
            self.argsort(X[:, j], idx_view)
            for i in range(0, n_samples):
                temp_int = idx_view[i]
                y_sort_view[i] = y[temp_int]

            for i in prange(1, n_samples - 1, nogil=True):
                Q_view[i, j] = self.score(y_sort_view, i)

        # Identify best split
        (thresh_i, feature) = self.argmin(Q_view)
        best_gini = Q_view[thresh_i, feature]

        # Sort samples by split feature
        self.argsort(X[:, feature], feat_idx_view)
        for i in range(0, n_samples):
            temp_int = feat_idx_view[i]

            # Sort X so we can get threshold
            feat_sort_view[i] = X[temp_int, feature]
            
            # Sort y so we can get left_y, right_y
            y_sort_view[i] = y[temp_int]
            
            # Sort true sample inds
            si_return_view[i] = sample_inds[temp_int]

        # Get threshold, split samples into left and right
        threshold = feat_sort[thresh_i]
        left_idx = si_return[:thresh_i]
        right_idx = si_return[thresh_i:]

        # Evaluate improvement
        improvement = node_impurity - best_gini

        # Evaluate impurities for left and right children
        left_impurity = self.impurity(y_sort_view[:thresh_i])
        right_impurity = self.impurity(y_sort_view[thresh_i:])

        return feature, threshold, left_impurity, left_idx, right_impurity, right_idx, improvement 

