cimport numpy as np
import numpy as np


cdef extern from "errno.h":
    int errno


cdef extern from "svmrank_parser.h":
    cdef struct shape:
        unsigned long cols
        unsigned long rows
    int PARSE_OK
    int PARSE_FILE_ERROR
    int PARSE_FORMAT_ERROR
    int PARSE_MEMORY_ERROR
    int c_parse_svmrank_file "parse_svmrank_file" (char* path, double** xs, shape* xs_shape, int** ys, long** qids) nogil
    void init_svmrank_parser()


def parse_svmrank_file(path):
    global errno

    # Initialize pointers
    cdef int* ys
    cdef long* qids
    cdef double* xs
    cdef shape xs_shape

    # Initialize path to read file from
    py_path_bytes = path.encode('UTF-8')
    cdef char* c_path = py_path_bytes
    cdef int result = 0

    # Initialize output array views
    cdef int[:] ys_view
    cdef long[:] qids_view
    cdef double[:,:] xs_view

    # Init parser and parse file
    init_svmrank_parser()
    with nogil:
        result = c_parse_svmrank_file(c_path, &xs, &xs_shape, &ys, &qids)

    if result == PARSE_OK:
        ys_view = <int[:xs_shape.rows]> ys
        ys_np = np.asarray(ys_view)
        qids_view = <long[:xs_shape.rows]> qids
        qids_np = np.asarray(qids_view)
        xs_view = <double[:xs_shape.rows,:xs_shape.cols]> xs
        xs_np = np.asarray(xs_view)

        return xs_np, ys_np, qids_np
    elif result == PARSE_FILE_ERROR:
        raise OSError(errno, "could not open file %s" % path)
    elif result == PARSE_FORMAT_ERROR:
        raise ValueError("could not parse file %s, not in SVMrank format" % path)
    elif result == PARSE_MEMORY_ERROR:
        raise OSError(errno, "could not allocate memory")
