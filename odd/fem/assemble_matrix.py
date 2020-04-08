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
from scipy.sparse import coo_matrix

# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)


def assemble_matrix(a, active_entities={}):
    ufc_form = dolfinx.jit.ffcx_jit(a)
    # noinspection PyProtectedMember
    _a = dolfinx.Form(a)._cpp_object
    mesh = _a.mesh()

    # Fixme: Allow different scalar types
    dtype = numpy.complex128

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    cell_map = mesh.topology.index_map(tdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts

    # Unpack geometry data
    x = mesh.geometry.x[:, :gdim]
    pos = mesh.geometry.dofmap().offsets()
    x_dofs = mesh.geometry.dofmap().array()
    nv = mesh.ufl_cell().num_vertices()

    # Unpack dofmap data
    dofmap0 = _a.function_space(0).dofmap
    dofmap1 = _a.function_space(1).dofmap
    assert dofmap0 == dofmap1
    ndofs_cell = dofmap0.dof_layout.num_dofs
    dof_array = dofmap0.list().array()
    N = numpy.max(dof_array) + 1

    data = numpy.zeros(ndofs_cell * dof_array.size, dtype=dtype)
    coefficients = dolfinx.cpp.fem.pack_coefficients(_a)
    constants = dolfinx.cpp.fem.pack_constants(_a)
    perm = numpy.array([0], dtype=numpy.uint8)

    if ufc_form.num_cell_integrals:
        active_cells = active_entities.get("cells", numpy.arange(num_cells))
        cell_integral = ufc_form.create_cell_integral(-1)
        kernel = cell_integral.tabulate_tensor
        assemble_cells(data, kernel, (dof_array, ndofs_cell), (pos, x_dofs, x, nv),
                       coefficients, constants, perm, active_cells)

    if ufc_form.num_exterior_facet_integrals:
        mesh.create_connectivity(tdim - 1, tdim)
        active_facets = active_entities.get("facets", numpy.where(mesh.topology.on_boundary(tdim - 1))[0])
        facet_data = facet_info(mesh, active_facets)
        facet_integral = ufc_form.create_exterior_facet_integral(-1)
        kernel = facet_integral.tabulate_tensor
        assemble_facets(data, kernel, (dof_array, ndofs_cell), (pos, x_dofs, x, nv),
                        coefficients, constants, perm, facet_data)

    rows, cols = sparsity_pattern(dof_array, ndofs_cell)
    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return A


@numba.njit(fastmath=True)
def assemble_cells(data, kernel, dofmap, mesh, coeffs, constants, perm, active_cells):
    (dof_array, ndofs_cell) = dofmap
    (pos, x_dofs, x, nv) = mesh

    entity_local_index = numpy.array([0], dtype=numpy.int32)
    local_mat = numpy.zeros((ndofs_cell, ndofs_cell), dtype=data.dtype)
    coordinate_dofs = numpy.zeros((nv, x.shape[1]), dtype=numpy.float64)
    for idx in active_cells:
        coordinate_dofs[:] = x[x_dofs[pos[idx]:pos[idx + 1]], :]
        local_mat.fill(0.0)
        kernel(ffi.from_buffer(local_mat), ffi.from_buffer(coeffs[idx, :]),
               ffi.from_buffer(constants), ffi.from_buffer(coordinate_dofs),
               ffi.from_buffer(entity_local_index), ffi.from_buffer(perm), 0)
        data[idx * local_mat.size:idx * local_mat.size + local_mat.size] += local_mat.ravel()


@numba.njit(fastmath=True)
def assemble_facets(data, kernel, dofmap, mesh, coeffs, constants, perm, facet_data):
    (dof_array, ndofs_cell) = dofmap
    (pos, x_dofs, x, nv) = mesh
    entity_local_index = numpy.array([0], dtype=numpy.int32)
    Ae = numpy.zeros((ndofs_cell, ndofs_cell), dtype=data.dtype)
    coordinate_dofs = numpy.zeros((nv, x.shape[1]), dtype=numpy.float64)
    nfacets = facet_data.shape[0]
    for i in range(nfacets):
        local_facet, cell_idx = facet_data[i]
        entity_local_index[0] = local_facet
        coordinate_dofs[:] = x[x_dofs[pos[cell_idx]:pos[cell_idx + 1]], :]
        Ae.fill(0.0)
        kernel(ffi.from_buffer(Ae), ffi.from_buffer(coeffs[cell_idx, :]),
               ffi.from_buffer(constants), ffi.from_buffer(coordinate_dofs),
               ffi.from_buffer(entity_local_index), ffi.from_buffer(perm), 0)
        data[cell_idx * Ae.size:cell_idx * Ae.size + Ae.size] += Ae.ravel()


def sparsity_pattern(dof_array, ndofs_cell):
    num_cells = dof_array.size // ndofs_cell
    rows = numpy.repeat(dof_array, ndofs_cell)
    cols = numpy.tile(numpy.reshape(dof_array, (num_cells, ndofs_cell)), ndofs_cell)
    return rows, cols.ravel()


def facet_info(mesh, active_facets):
    # get facet-cell and cell-facet connections
    tdim = mesh.topology.dim
    c2f = mesh.topology.connectivity(tdim, tdim - 1).array()
    c2f_offsets = mesh.topology.connectivity(tdim, tdim - 1).offsets()
    f2c = mesh.topology.connectivity(tdim - 1, tdim).array()
    f2c_offsets = mesh.topology.connectivity(tdim - 1, tdim).offsets()
    cell_map = mesh.topology.index_map(tdim)
    num_cells = cell_map.size_local + cell_map.num_ghosts

    @numba.njit(fastmath=True, cache=True)
    def facet2cell():
        facet_data = numpy.zeros((active_facets.size, 2), dtype=numpy.int32)
        for j, facet in enumerate(active_facets):
            cells = f2c[f2c_offsets[facet]:f2c_offsets[facet + 1]]
            local_facets = c2f[c2f_offsets[cells[0]]:c2f_offsets[cells[0] + 1]]
            local_facet = numpy.where(facet == local_facets)[0][0]
            facet_data[j, 0] = local_facet
            facet_data[j, 1] = cells[0]
        return facet_data

    return facet2cell()
