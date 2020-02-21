# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from petsc4py import PETSc
import dolfinx
import ufl

_create_cpp_form = dolfinx.fem.assemble._create_cpp_form


def assemble_vector(L: ufl.Form) -> PETSc.Vec:
    '''
    Create and assemble vector given a rank 1 ufl form
    '''
    _L = _create_cpp_form(L)
    if _L.rank != 1:
        raise ValueError

    b = dolfinx.fem.create_vector(_L)
    dolfinx.cpp.fem.assemble_vector(b, _L)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    return b
