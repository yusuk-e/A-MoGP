# -*- coding:utf-8 -*-

import numpy as np
cimport numpy as np
from time import time
from libc.math cimport log
from libc.math cimport exp
from libc.math cimport trunc

DTYPE = np.double
ctypedef np.double_t DTYPE_t

DTYPE2 = np.int
ctypedef np.int_t DTYPE2_t

def make_norm(np.ndarray[DTYPE_t, ndim=2] grids):

    cdef np.ndarray[DTYPE_t, ndim=1] norms = np.zeros(1)
    cdef double x1,y1,x2,y2
    cdef double norm,tmp
    cdef int flag,i,j,k

    for i in range(len(grids)):
        x1 = grids[i,0]
        y1 = grids[i,1]
        for j in range(len(grids)):
            x2 = grids[j,0]
            y2 = grids[j,1]
            norm = (x1-x2) ** 2 + (y1-y2) ** 2
            norm = trunc(norm*100)/100

            flag = 0
            for k in range(len(norms)):
                tmp = norms[k]
                if norm - tmp == 0:
                    flag = 1
                    break
            if flag == 0:
                norms = np.hstack([norms,norm])

    return norms


def make_rr_count(np.ndarray[DTYPE_t, ndim=2] grids, np.ndarray[DTYPE2_t, ndim=1] grid_ids1, np.ndarray[DTYPE2_t, ndim=1] grid_ids2, np.ndarray[DTYPE_t, ndim=1] norms):

    cdef double x1,y1,x2,y2
    cdef double norm,tmp
    cdef int i,j,k,N,grid_id1,grid_id2,count_id
    N = np.size(norms)
    cdef np.ndarray[DTYPE2_t, ndim=1] count = np.zeros(N).astype(int)

    for i in range(len(grid_ids1)):
        grid_id1 = int(grid_ids1[i])
        x1 = grids[grid_id1,0]
        y1 = grids[grid_id1,1]
        for j in range(len(grid_ids2)):
            grid_id2 = int(grid_ids2[j])
            x2 = grids[grid_id2,0]
            y2 = grids[grid_id2,1]
            norm = (x1-x2)**2 + (y1-y2)**2
            norm = trunc(norm*100)/100

            count_id = -1
            for k in range(len(norms)):
                tmp = norms[k]
                if norm - tmp == 0:
                    count_id = k
                    break
            count[count_id] += 1

    return count


def make_rp_count(np.ndarray[DTYPE_t, ndim=2] grids, np.ndarray[DTYPE_t, ndim=1] grid_pred, np.ndarray[DTYPE2_t, ndim=1] grid_ids, np.ndarray[DTYPE_t, ndim=1] norms):

    cdef double x_pred,y_pred,x,y
    cdef double norm,tmp
    cdef int i,j,k,N,grid_id,count_id
    N = np.size(norms)
    cdef np.ndarray[DTYPE2_t, ndim=1] count = np.zeros(N).astype(int)

    x_pred = grid_pred[0]
    y_pred = grid_pred[1]
    for i in range(len(grid_ids)):
        grid_id = int(grid_ids[i])
        x = grids[grid_id,0]
        y = grids[grid_id,1]
        norm = (x_pred-x)**2 + (y_pred-y)**2
        norm = trunc(norm*100)/100

        count_id = -1
        for k in range(len(norms)):
            tmp = norms[k]
            if norm - tmp == 0:
                count_id = k
                break
        count[count_id] += 1

    return count
