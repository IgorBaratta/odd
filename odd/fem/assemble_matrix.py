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

from .mesh import mesh_wrapper, MeshWrapper
from .dofmap import dofmap_wrapper, DofMapWrapper

# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof("double _Complex"), numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof("float _Complex"), numba.types.complex64)


def assemble_matrix(a, active_entities=None):

    if active_entities is None:
        active_entities = {}

    # Todo: Allow different types
    dtype = numpy.complex128

    ufc_form = dolfinx.jit.ffcx_jit(a)
    _a = dolfinx.Form(a)._cpp_object
    mesh = mesh_wrapper(_a.mesh())
    dofmap = dofmap_wrapper(_a.function_space(0).dofmap)

    data = numpy.zeros(dofmap.num_cell_dofs * dofmap.dof_array.size, dtype=dtype)
    coefficients = dolfinx.cpp.fem.pack_coefficients(_a)
    constants = dolfinx.cpp.fem.pack_constants(_a)
    perm = numpy.array([0], dtype=numpy.uint8)

    if ufc_form.num_cell_integrals:
        active_cells = active_entities.get("cells", numpy.arange(mesh.topology.num_cells))
        cell_integral = ufc_form.create_cell_integral(-1)
        kernel = cell_integral.tabulate_tensor
        assemble_cells(
            data, kernel, dofmap, mesh, coefficients, constants, perm, active_cells,
        )

    if ufc_form.num_exterior_facet_integrals:
        active_facets = active_entities.get("facets", mesh.topology.boundary_facets)
        facet_data = facet_info(_a.mesh(), active_facets)
        facet_integral = ufc_form.create_exterior_facet_integral(-1)
        kernel = facet_integral.tabulate_tensor
        assemble_facets(
            data, kernel, dofmap, mesh, coefficients, constants, perm, facet_data,
        )

    local_mat = coo_matrix((data, sparsity_pattern(dofmap)), shape=(dofmap.size, dofmap.size)).tocsr()
    return local_mat


@numba.njit(fastmath=True)
def assemble_cells(data, kernel, dofmap: DofMapWrapper, mesh: MeshWrapper, coeffs, constants, perm, active_cells):
    (dim, x, x_dofs, pos) = mesh.geometry
    entity_local_index = numpy.array([0], dtype=numpy.int32)
    local_mat = numpy.zeros((dofmap.num_cell_dofs, dofmap.num_cell_dofs), dtype=data.dtype)
    for idx in active_cells:
        coordinate_dofs = x[x_dofs[pos[idx] : pos[idx + 1]], :]
        local_mat.fill(0.0)
        kernel(
            ffi.from_buffer(local_mat),
            ffi.from_buffer(coeffs[idx, :]),
            ffi.from_buffer(constants),
            ffi.from_buffer(coordinate_dofs),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
            0,
        )
        data[idx * local_mat.size : idx * local_mat.size + local_mat.size] += local_mat.ravel()


@numba.njit(fastmath=True)
def assemble_facets(data, kernel, dofmap: DofMapWrapper, mesh: MeshWrapper, coeffs, constants, perm, facet_data):
    entity_local_index = numpy.array([0], dtype=numpy.int32)
    Ae = numpy.zeros((dofmap.num_cell_dofs, dofmap.num_cell_dofs), dtype=data.dtype)
    gdim, x, x_dofs, pos = mesh.geometry
    num_active_facets = facet_data.shape[0]
    for i in range(num_active_facets):
        local_facet, cell_idx = facet_data[i]
        entity_local_index[0] = local_facet
        coordinate_dofs = x[x_dofs[pos[cell_idx] : pos[cell_idx + 1]], :]
        Ae.fill(0.0)
        kernel(
            ffi.from_buffer(Ae),
            ffi.from_buffer(coeffs[cell_idx, :]),
            ffi.from_buffer(constants),
            ffi.from_buffer(coordinate_dofs),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
            0,
        )
        data[cell_idx * Ae.size : cell_idx * Ae.size + Ae.size] += Ae.ravel()


def sparsity_pattern(dofmap: DofMapWrapper):
    """
    Returns local COO sparsity pattern
    """
    num_cells = dofmap.dof_array.size // dofmap.num_cell_dofs
    rows = numpy.repeat(dofmap.dof_array, dofmap.num_cell_dofs)
    cols = numpy.tile(numpy.reshape(dofmap.dof_array, (num_cells, dofmap.num_cell_dofs)), dofmap.num_cell_dofs)
    return rows, cols.ravel()


def facet_info(mesh, active_facets):
    # FIXME: Refactor this function using the wrapper
    # get facet-cell and cell-facet connections
    tdim = mesh.topology.dim
    c2f = mesh.topology.connectivity(tdim, tdim - 1).array()
    c2f_offsets = mesh.topology.connectivity(tdim, tdim - 1).offsets()
    f2c = mesh.topology.connectivity(tdim - 1, tdim).array()
    f2c_offsets = mesh.topology.connectivity(tdim - 1, tdim).offsets()
    facet_data = numpy.zeros((active_facets.size, 2), dtype=numpy.int32)

    @numba.njit(fastmath=True)
    def facet2cell(data):
        for j, facet in enumerate(active_facets):
            cells = f2c[f2c_offsets[facet] : f2c_offsets[facet + 1]]
            local_facets = c2f[c2f_offsets[cells[0]] : c2f_offsets[cells[0] + 1]]
            local_facet = numpy.where(facet == local_facets)[0][0]
            data[j, 0] = local_facet
            data[j, 1] = cells[0]
        return data

    return facet2cell(facet_data)
