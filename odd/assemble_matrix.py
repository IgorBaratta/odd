# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import cffi
import dolfinx
import numpy
import numba
import numba.cffi_support
from mpi4py import MPI
from petsc4py import PETSc
from .petsc_utils import MatSetValues, MatSetValuesLocal
from .subdomain import on_interface


ffi = cffi.FFI()
_create_cpp_form = dolfinx.fem.assemble._create_cpp_form


def create_matrix(a, type="communication-less"):
    _a = _create_cpp_form(a)

    if type == "standard":
        A = dolfinx.cpp.fem.create_matrix(_a)
        A.zeroEntries()

    elif type == "communication-less":
        dofmap0 = _a.function_space(0).dofmap
        dofmap1 = _a.function_space(1).dofmap

        assert dofmap0 == dofmap1

        dof_array = dofmap0.list().array()
        ndofs_cell = dofmap0.dof_layout.num_dofs

        # Number of nonzeros per row
        pattern, nnz = sparsity_pattern(dof_array, ndofs_cell)
        ndofs = len(pattern)

        A = PETSc.Mat().createAIJ([ndofs, ndofs], nnz=nnz, comm=MPI.COMM_SELF)
        A.setUp()
        A.zeroEntries()

    elif type == "scipy":
        dofmap0 = _a.function_space(0).dofmap
        dof_array = dofmap0.list().array()
        ndofs_cell = dofmap0.dof_layout.num_dofs
        A = sparsity_pattern_scipy(dof_array, ndofs_cell)

    return A


@numba.njit(fastmath=True)
def sparsity_pattern(dof_array, ndofs_cell):
    '''
    Create the sparsity pattern of the matrix.
    Based on cell integral pattern.
    '''
    num_cells = int(dof_array.size/ndofs_cell)
    ndofs = numpy.max(dof_array) + 1
    pattern = [set([i]) for i in range(ndofs)]

    for cell in range(num_cells):
        cell_dof = dof_array[cell*ndofs_cell: cell*ndofs_cell + ndofs_cell]
        for dof0 in cell_dof:
            for dof1 in cell_dof:
                pattern[dof0].add(dof1)

    nnz = numpy.zeros(ndofs, dtype=numpy.int32)
    for i in range(ndofs):
        nnz[i] = len(pattern[i])

    return pattern, nnz


@numba.njit(fastmath=True)
def sparsity_pattern_scipy(dof_array, ndofs_cell):
    '''
    Create the sparsity pattern of the matrix.
    Based on cell integral pattern.
    '''
    num_cells = int(dof_array.size/ndofs_cell)

    vsize = num_cells * ndofs_cell**2

    rows = numpy.zeros(vsize, dtype=numpy.int32)
    cols = numpy.zeros(vsize, dtype=numpy.int32)
    vals = numpy.zeros(vsize, dtype=numpy.int32)

    j = 0
    for cell in range(num_cells):
        cell_dof = dof_array[cell*ndofs_cell: cell*ndofs_cell + ndofs_cell]
        for dof in cell_dof:
                rows[j: j + ndofs_cell] = dof
                cols[j: j + ndofs_cell] = cell_dof
                j += ndofs_cell
    return rows, cols, vals


def assemble_matrix(a, A, active_entities={}):
    ufc_form = dolfinx.jit.ffcx_jit(a)
    _a = _create_cpp_form(a)
    mesh = _a.mesh()

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    num_cells = mesh.num_entities(tdim)

    dofmap0 = _a.function_space(0).dofmap
    dofmap1 = _a.function_space(1).dofmap
    assert dofmap0 == dofmap1

    # Unpack mesh and dofmap data
    x = mesh.geometry.x[:, :gdim]
    pos = mesh.geometry.dofmap().offsets()
    x_dofs = mesh.geometry.dofmap().array()

    # Unpack dofmap data
    ndofs_cell = dofmap0.dof_layout.num_dofs
    dof_array = dofmap0.list().array()

    if isinstance(A, PETSc.Mat):
        mat = A.handle
        if A.type == 'seqaij':
            set_values = MatSetValues
        elif A.type == 'mpiaij':
            set_values = MatSetValuesLocal

    insert_mode = PETSc.InsertMode.ADD

    if ufc_form.num_cell_integrals:
        active_cells = active_entities.get("cells", numpy.arange(num_cells))
        cell_integral = ufc_form.create_cell_integral(-1)
        kernel = cell_integral.tabulate_tensor
        assemble_cells(mat, kernel, (dof_array, ndofs_cell), (pos, x_dofs, x),
                       active_cells, set_values, insert_mode)


@numba.njit(fastmath=True)
def assemble_cells(mat, kernel, dofmap, mesh, active_cells, set_values, insert_mode):
    (dof_array, ndofs_cell) = dofmap
    (pos, x_dofs, x) = mesh

    Ae = numpy.zeros((ndofs_cell, ndofs_cell), dtype=PETSc.ScalarType)
    coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
    constants = numpy.zeros(1, dtype=PETSc.ScalarType)

    entity_local_index = numpy.array([0], dtype=numpy.int32)
    perm = numpy.array([0], dtype=numpy.uint8)

    coordinate_dofs = numpy.zeros((ndofs_cell, x.shape[1]), dtype=PETSc.RealType)

    for idx in active_cells:
        coordinate_dofs = x[x_dofs[pos[idx]:pos[idx+1]], :]
        Ae.fill(0.0)
        kernel(ffi.from_buffer(Ae), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants), ffi.from_buffer(coordinate_dofs),
               ffi.from_buffer(entity_local_index), ffi.from_buffer(perm), 0)

        rows = dof_array[idx*ndofs_cell: idx*ndofs_cell + ndofs_cell]
        cols = dof_array[idx*ndofs_cell: idx*ndofs_cell + ndofs_cell]

        set_values(mat, ndofs_cell, ffi.from_buffer(rows),
                   ndofs_cell, ffi.from_buffer(cols),
                   ffi.from_buffer(Ae), insert_mode)


def apply_transmission_condition(A, s):
    _s = _create_cpp_form(s)
    active_facets = numpy.where(on_interface(_s.mesh()))[0]
    assemble_matrix(s, A, {"facets": active_facets})
