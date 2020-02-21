# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
from petsc4py import PETSc


def apply_bc(ndarray, dofs, values=None):
    if isinstance(ndarray, PETSc.Mat):
        _matrix_apply_bc(ndarray, dofs)
    elif isinstance(ndarray, PETSc.Vec):
        _vector_apply_bc(ndarray, dofs, values)
    else:
        raise TypeError


def _matrix_apply_bc(A, dofs):
    A.zeroRows(rows=dofs.flatten(), diag=1)


def _vector_apply_bc(b, dofs, values):
    assert(dofs.size == values.size)
    with b.localForm() as b:
        b.setValues(dofs.astype(PETSc.IntType), values.astype(PETSc.ScalarType))
