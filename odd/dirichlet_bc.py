# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import scipy
import numpy


def apply_bc(data, dofs, values=None):
    '''
    Simplified interface for applying Dirichlet BC.

    '''
    if isinstance(data, scipy.sparse.spmat):
        _matrix_apply_bc(data, dofs)
        data.eliminate_zeros()
    elif isinstance(data, numpy.ndarray):
        _vector_apply_bc(data, dofs, values)
    else:
        raise TypeError


def _matrix_apply_bc(A, dofs):
    A.zeroRows(rows=dofs.flatten(), diag=1)


def _vector_apply_bc(b, dofs, values):
    b[dofs] = values


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError('Matrix given must be of CSR format.')
    csr.data[csr.indptr[row]:csr.indptr[row+1]] = value
