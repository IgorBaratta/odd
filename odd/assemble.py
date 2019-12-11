# Copyright (C) 2019 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin
import ufl
import numpy
import numba
import numba.cffi_support
from petsc4py import PETSc
import cffi


# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)


def assemble_vector(L: ufl.Form):
    """
    Assemble linear form into a new PETSc vector.
    """
    ufc_form = dolfin.jit.ffc_jit(L)
    V = dolfin.fem.assemble._create_cpp_form(L).function_space(0)
    b = dolfin.cpp.la.create_vector(V.dofmap.index_map)
    dim = V.mesh.geometry.dim
    geom = V.mesh.geometry.points
    dofs = V.dofmap.dof_array()
    dofs_per_cell = V.dofmap.dof_layout.num_dofs

    with b.localForm() as b_local:
        b_local.set(0.0)
        if ufc_form.num_custom_integrals > 0:
            raise NotImplementedError
        if ufc_form.num_exterior_facet_integrals > 0:
            kernel = ufc_form.create_exterior_facet_integral(-1).tabulate_tensor
            c = V.mesh.topology.connectivity(dim, 0).connections()
        if ufc_form.num_interior_facet_integrals > 0:
            raise NotImplementedError
        if ufc_form.num_cell_integrals > 0:
            kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
            c = V.mesh.topology.connectivity(dim, 0).connections()
            pos = V.mesh.topology.connectivity(dim, 0).pos()
            assemble_vector_kernel(numpy.asarray(b), kernel, c, pos, geom, dofs, dofs_per_cell)

    return b


@numba.njit
def assemble_vector_kernel(b, kernel, connections, positions, x, dofmap, dofs_per_cell):
    """Communicationless assemble provided FFC/UFC kernel over a
    mesh into the array b"""

    orientation = numpy.array([0], dtype=numpy.int32)
    coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
    constants = numpy.zeros(1, dtype=PETSc.ScalarType)
    geometry = numpy.zeros((3, 2))
    b_local = numpy.zeros(dofs_per_cell, dtype=PETSc.ScalarType)
    for i, cell in enumerate(positions[:-1]):
        num_vertices = positions[i + 1] - positions[i]
        c = connections[cell:cell + num_vertices]
        for j in range(3):
            for k in range(2):
                geometry[j, k] = x[c[j], k]
        b_local.fill(0.0)
        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants),
               ffi.from_buffer(geometry), ffi.from_buffer(orientation),
               ffi.from_buffer(orientation))
        for j in range(dofs_per_cell):
            b[dofmap[i * dofs_per_cell + j]] += b_local[j]
