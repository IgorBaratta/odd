# Copyright (C) 2020 Igor A. Baratta
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
from petsc_utils import MatSetValues_ctypes as MatSetValues


def _create_cpp_form(form):
    """Recursively look for ufl.Forms and convert to dolfinx.fem.Form, otherwise
    return form argument
    """
    if isinstance(form, dolfinx.Form):
        return form._cpp_object
    elif isinstance(form, ufl.Form):
        return dolfinx.Form(form)._cpp_object
    elif isinstance(form, (tuple, list)):
        raise NotImplementedError
    return form


# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)


def assemble_matrix(a):
    _a = _create_cpp_form(a)
    A = dolfinx.cpp.fem.create_matrix(_a)
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

    # @numba.njit
    def assemble_cells(kernel):
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
            # A.setValuesLocal(rows, cols, Ae, PETSc.InsertMode.ADD)
            MatSetValues(A.handle, num_dofs_per_cell0, rows.ctypes, num_dofs_per_cell1, cols.ctypes,
                         Ae.ctypes, PETSc.InsertMode.ADD)

    # @numba.jit
    def assemble_facets(kernel):
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
            A.setValuesLocal(rows, cols, Ae, PETSc.InsertMode.ADD)

    if num_cell_int:
        cell_integral = ufc_form.create_cell_integral(-1)
        kernel = cell_integral.tabulate_tensor
        assemble_cells(kernel)
    if num_facet_int:
        facet_integral = ufc_form.create_exterior_facet_integral(-1)
        kernel = facet_integral.tabulate_tensor
        assemble_facets(kernel)

    return A


mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_self, 100, 100)
V = dolfinx.FunctionSpace(mesh, ("Lagrange", 3))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
k = 10 * numpy.pi

a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + k * ufl.inner(u, v)*ufl.dx +\
    1j * ufl.inner(u, v) * ufl.ds

A = assemble_matrix(a)
A.assemble()
B = dolfinx.fem.assemble_matrix(a)
B.assemble()

print((A-B).norm())
