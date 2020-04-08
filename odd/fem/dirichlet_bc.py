# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
import scipy
import numpy


def apply_bc(data, dof_list, values=None):
    """
    Simplified interface for applying Dirichlet BC.

    """
    dofs = dof_list.ravel()
    if isinstance(data, scipy.sparse.spmatrix):
        _matrix_apply_bc(data, dofs)
    elif isinstance(data, numpy.ndarray):
        _vector_apply_bc(data, dofs, values)
    else:
        raise TypeError


def _matrix_apply_bc(A, dofs):
    if not isinstance(A, scipy.sparse.csr_matrix):
        raise ValueError("Matrix must be of csr format.")
    for dof in dofs:
        A.data[A.indptr[dof] : A.indptr[dof + 1]] = 0.0
        A[dof, dof] = 1.0


def _vector_apply_bc(b, dofs, values):
    assert dofs.size == values.size
    b[dofs] = values
