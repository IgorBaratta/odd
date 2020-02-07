# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import numpy
import numba
import numba.cffi_support
from mpi4py import MPI
from petsc4py import PETSc
import cffi
from .petsc_utils import MatSetValues, MatSetValuesLocal


_create_cpp_form = dolfinx.fem.assemble._create_cpp_form

# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)


def create_matrix(a, type="standard"):
    _a = _create_cpp_form(a)

    if type == "standard":
        A = dolfinx.cpp.fem.create_matrix(_a)
        A.zeroEntries()
        return A

    elif type == "communication-less":
        dofmap0 = _a.function_space(0).dofmap
        dofmap1 = _a.function_space(1).dofmap

        dof_array0 = dofmap0.dof_array()
        dof_array1 = dofmap1.dof_array()

        num_dofs_per_cell0 = dofmap0.dof_layout.num_dofs
        num_dofs_per_cell1 = dofmap1.dof_layout.num_dofs

        # Number of nonzeros per row
        nnz = sparsity_pattern((dof_array0, num_dofs_per_cell0),
                               (dof_array1, num_dofs_per_cell1))
        size = nnz.size
        A = PETSc.Mat().createAIJ([size, size], nnz=nnz, comm=MPI.COMM_SELF)
        A.setUp()
        A.zeroEntries()

        return A


@numba.njit
def sparsity_pattern(dofmap0, dofmap1):
    '''
    Return an estimated number of non zeros per row.
    '''
    # TODO: improve sparsity pattern
    # Based on cell integral pattern
    (dof_array0, ndofs0) = dofmap0
    (dof_array1, ndofs1) = dofmap1
    num_cells = int(dof_array0.size/ndofs0)

    num_dofs = numpy.unique(dof_array0)
    pattern = numpy.zeros_like(num_dofs)

    for cell_index in range(num_cells):
        cell_dof0 = dof_array0[cell_index*ndofs0: cell_index*ndofs0 + ndofs0]
        for dof0 in cell_dof0:
                pattern[dof0] = pattern[dof0] + ndofs1
    return pattern


def assemble_matrix(a, A):
    _a = _create_cpp_form(a)
    inssert_mode = PETSc.InsertMode.ADD
    ufc_form = dolfinx.jit.ffcx_jit(a)
    mesh = _a.mesh()

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    num_cells = mesh.num_entities(tdim)
    points = mesh.geometry.points

    dofmap0 = _a.function_space(0).dofmap
    dofmap1 = _a.function_space(1).dofmap
    dof_array0 = dofmap0.dof_array()
    dof_array1 = dofmap1.dof_array()

    num_dofs_per_cell0 = dofmap0.dof_layout.num_dofs
    num_dofs_per_cell1 = dofmap1.dof_layout.num_dofs

    num_cell_int = ufc_form.num_cell_integrals
    num_facet_int = ufc_form.num_exterior_facet_integrals

    # Start connections
    connectivity = mesh.topology.connectivity
    cell_g = connectivity(tdim, 0).connections()
    pos_g = connectivity(tdim, 0).pos()
    num_dofs_g = connectivity(tdim, 0).size(0)

    if num_facet_int:
        facet_cell = mesh.topology.connectivity(tdim - 1, tdim).connections()
        pos_facet = mesh.topology.connectivity(tdim - 1, tdim).pos()
        cell_facet = mesh.topology.connectivity(tdim, tdim - 1).connections()
        pos_cell = mesh.topology.connectivity(tdim, tdim - 1).pos()
        facets_on_boundary = mesh.topology.on_boundary(tdim - 1)
        active_facets = numpy.where(facets_on_boundary)[0]

    @numba.njit(cache=False)
    def assemble_cells(kernel, mat, set_values):
        # Cannot cache compiled function "assemble_cells" as it uses outer variables in a closure
        active_cells = numpy.arange(num_cells)
        Ae = numpy.zeros((num_dofs_per_cell0, num_dofs_per_cell1), dtype=PETSc.ScalarType)
        orientation = numpy.array([0], dtype=PETSc.IntType)
        coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
        constants = numpy.zeros(1, dtype=PETSc.ScalarType)
        coordinate_dofs = numpy.zeros((num_dofs_g, gdim), dtype=PETSc.RealType)
        for cell_index in active_cells:
            # Get cell coordinates/geometry
            for i in range(num_dofs_g):
                for j in range(gdim):
                    coordinate_dofs[i, j] = points[cell_g[pos_g[cell_index] + i], j]

            Ae.fill(0.0)
            kernel(ffi.from_buffer(Ae), ffi.from_buffer(coeffs),
                   ffi.from_buffer(constants),
                   ffi.from_buffer(coordinate_dofs), ffi.from_buffer(orientation),
                   ffi.from_buffer(orientation))

            rows = dof_array0[cell_index*num_dofs_per_cell0:
                              cell_index*num_dofs_per_cell0 + num_dofs_per_cell0]
            cols = dof_array1[cell_index*num_dofs_per_cell1:
                              cell_index*num_dofs_per_cell1 + num_dofs_per_cell1]

            set_values(mat, num_dofs_per_cell0, ffi.from_buffer(rows),
                       num_dofs_per_cell1, ffi.from_buffer(cols),
                       ffi.from_buffer(Ae), inssert_mode)

    @numba.njit(cache=False)
    def assemble_facets(kernel, mat, set_values):
        # Cannot cache compiled function "assemble_facets" as it uses outer variables in a closure
        Ae = numpy.zeros((num_dofs_per_cell0, num_dofs_per_cell1), dtype=PETSc.ScalarType)
        orientation = numpy.array([0], dtype=PETSc.IntType)
        coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
        constants = numpy.zeros(1, dtype=PETSc.ScalarType)
        coordinate_dofs = numpy.zeros((num_dofs_g, gdim), dtype=PETSc.RealType)

        for facet_index in active_facets:
            cell_index = facet_cell[pos_facet[facet_index]]

            # Get local index of facet with respect to the cell
            facets = cell_facet[pos_cell[cell_index]: pos_cell[cell_index + 1]]
            local_facet = numpy.where(facets == facet_index)[0].astype(PETSc.IntType)

            # Get cell coordinates/geometry
            for i in range(num_dofs_g):
                for j in range(gdim):
                    coordinate_dofs[i, j] = points[cell_g[pos_g[cell_index] + i], j]

            Ae.fill(0.0)
            kernel(ffi.from_buffer(Ae), ffi.from_buffer(coeffs),
                   ffi.from_buffer(constants),
                   ffi.from_buffer(coordinate_dofs), ffi.from_buffer(local_facet),
                   ffi.from_buffer(orientation))

            rows = dof_array0[cell_index*num_dofs_per_cell0:
                              cell_index*num_dofs_per_cell0 + num_dofs_per_cell0]
            cols = dof_array1[cell_index*num_dofs_per_cell1:
                              cell_index*num_dofs_per_cell1 + num_dofs_per_cell1]

            set_values(mat, num_dofs_per_cell0, ffi.from_buffer(rows),
                       num_dofs_per_cell1, ffi.from_buffer(cols),
                       ffi.from_buffer(Ae), inssert_mode)

    if A.type == 'seqaij':
        set_values = MatSetValues
    elif A.type == 'mpiaij':
        set_values = MatSetValuesLocal

    if num_cell_int:
        cell_integral = ufc_form.create_cell_integral(-1)
        kernel = cell_integral.tabulate_tensor
        mat = A.handle
        assemble_cells(kernel, mat, set_values)
    if num_facet_int:
        facet_integral = ufc_form.create_exterior_facet_integral(-1)
        kernel = facet_integral.tabulate_tensor
        mat = A.handle
        assemble_facets(kernel, mat, set_values)
    return A
