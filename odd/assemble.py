# Copyright (C) 2019 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
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


def assemble_vector(L: ufl.Form)->PETSc.Vec:
    """
    Communicationless assemble provided FFC/UFC kernel over a mesh into the array b
    """
    ufc_form = dolfinx.jit.ffc_jit(L)
    V = dolfinx.fem.assemble._create_cpp_form(L).function_space(0)
    b = dolfinx.cpp.la.create_vector(V.dofmap.index_map)
    dim = V.mesh.geometry.dim
    points = V.mesh.geometry.points
    dofs = V.dofmap.dof_array()
    dofs_per_cell = V.dofmap.dof_layout.num_dofs

    with b.localForm() as b_local:
        b_local.set(0.0)

        if ufc_form.num_coefficients:
            raise NotImplementedError

        if ufc_form.num_custom_integrals:
            raise NotImplementedError

        if ufc_form.num_interior_facet_integrals:
            raise NotImplementedError

        if ufc_form.num_exterior_facet_integrals > 0:
            print("assembling exterior facet integrals")
            kernel = ufc_form.create_exterior_facet_integral(-1).tabulate_tensor

            # Prepare connectivities cv (cell vertex), fc (facet cell), cf(cell facet)
            c_cv = V.mesh.topology.connectivity(dim, 0).connections()
            c_fc = V.mesh.topology.connectivity(dim - 1, dim).connections()
            c_cf = V.mesh.topology.connectivity(dim, dim - 1).connections()
            pos_cv = V.mesh.topology.connectivity(dim, 0).pos()
            pos_fc = V.mesh.topology.connectivity(dim - 1, dim).pos()
            pos_cf = V.mesh.topology.connectivity(dim, dim - 1).pos()
            connectivities = (c_cv, c_fc, c_cf, pos_cv, pos_fc, pos_cf)
            on_boundary = V.mesh.topology.on_boundary(dim - 1)
            active_facets = numpy.where(on_boundary)[0]
            assemble_exterior_facets(numpy.asarray(b), kernel, connectivities,
                                     points, dofs, dofs_per_cell, active_facets)
        if ufc_form.num_cell_integrals > 0:
            kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
            c = V.mesh.topology.connectivity(dim, 0).connections()
            pos = V.mesh.topology.connectivity(dim, 0).pos()
            assemble_cells(numpy.asarray(b), kernel, (c, pos), points, dofs, dofs_per_cell)

    return b


@numba.njit
def assemble_cells(b, kernel, mesh, x, dofmap, dofs_per_cell):
    """Execute kernel over cells and accumulate result in vector"""
    connections, positions = mesh
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


# @numba.njit
def assemble_exterior_facets(b, kernel, connectivities, x, dofmap,
                             dofs_per_cell, active_facets):
    """Execute kernel over cells and accumulate result in vector"""
    orientation = numpy.array([0], dtype=numpy.int32)
    coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
    constants = numpy.zeros(1, dtype=PETSc.ScalarType)
    geometry = numpy.zeros((3, 2), dtype=numpy.float64)
    b_local = numpy.zeros(dofs_per_cell, dtype=PETSc.ScalarType)
    (c_cv, c_fc, c_cf, pos_cv, pos_fc, pos_cf) = connectivities
    for facet in active_facets:
        cell = c_fc[pos_fc[facet]]
        num_facets = pos_cf[cell + 1] - pos_cf[cell]
        num_vertices = pos_cv[cell + 1] - pos_cv[cell]
        facets = c_cf[pos_cf[cell]: pos_cf[cell] + num_facets]
        local_facet = numpy.where(facets == facet)[0]
        local_facet = local_facet.astype(numpy.int32)
        vertices = c_cv[pos_cv[cell]:pos_cv[cell] + num_vertices]
        for j in range(3):
            for k in range(2):
                geometry[j, k] = x[vertices[j], k]
                print(geometry)
        b_local.fill(0.0)
        kernel(ffi.from_buffer(b_local), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants), ffi.from_buffer(geometry),
               ffi.from_buffer(local_facet), ffi.from_buffer(orientation))
        for j in range(dofs_per_cell):
            b[dofmap[cell * dofs_per_cell + j]] += b_local[j]
