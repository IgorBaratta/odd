# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from petsc4py import PETSc
import dolfinx
import numpy
import ufl


def assemble_vector(form: ufl.Form, dtype=numpy.complex128) -> numpy.ndarray:
    """
    Create and assemble vector given a rank 1 ufl form
    """
    _L = dolfinx.Form(form)._cpp_object
    if _L.rank != 1:
        raise ValueError

    dofmap = _L.function_space(0).dofmap

    b = dolfinx.fem.create_vector(_L)
    dolfinx.cpp.fem.assemble_vector(b, _L)
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    vec_size = dofmap.index_map.size_local + dofmap.index_map.num_ghosts

    np_b = numpy.zeros(vec_size, dtype)
    with b.localForm() as b_local:
        if not b_local:
            np_b = b.array
        else:
            np_b = b_local.array
    return np_b
