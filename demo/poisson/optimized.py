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


from dolfinx.common import Timer, list_timings, TimingType

# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)


def _create_cpp_form(form):
    """Recursively look for ufl.Forms and convert to dolfinx.fem.Form, otherwise
    return form argument
    """
    if isinstance(form, dolfinx.Form):
        return form._cpp_object
    elif isinstance(form, ufl.Form):
        return dolfinx.Form(form)._cpp_object
    elif isinstance(form, (tuple, list)):
        return list(map(lambda sub_form: _create_cpp_form(sub_form), form))
    return form


mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_self, 1, 1)

V = dolfinx.FunctionSpace(mesh, ("Lagrange", 5))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = 1j * ufl.inner(u, v) * ufl.ds

_a = _create_cpp_form(a)
A = dolfinx.cpp.fem.create_matrix(_a)

ufc_form = dolfinx.jit.ffcx_jit(a)

mesh = _a.mesh()
tdim = mesh.topology.dim
gdim = mesh.geometry.dim

dofmap0 = _a.function_space(0).dofmap
dofmap1 = _a.function_space(1).dofmap
dof_array0 = dofmap0.dof_array()
dof_array1 = dofmap1.dof_array()

num_dofs_per_cell0 = dofmap0.dof_layout.num_dofs
num_dofs_per_cell1 = dofmap1.dof_layout.num_dofs

if ufc_form.num_exterior_facet_integrals == 1:
    facets_on_boundary = mesh.topology.on_boundary(tdim - 1)
    active_facets = numpy.where(facets_on_boundary)[0]
    facet_integral = ufc_form.create_exterior_facet_integral(-1)
    kernel = facet_integral.tabulate_tensor

    cell_g = mesh.topology.connectivity(tdim, 0).connections()
    pos_g = mesh.topology.connectivity(tdim, 0).pos()
    x_g = mesh.geometry.points
    num_dofs_g = mesh.topology.connectivity(tdim, 0).size(0)

    facet_cell = mesh.topology.connectivity(tdim - 1, tdim).connections()
    pos_facet = mesh.topology.connectivity(tdim - 1, tdim).pos()

    cell_facet = mesh.topology.connectivity(tdim, tdim - 1).connections()
    pos_cell = mesh.topology.connectivity(tdim, tdim - 1).pos()

    Ae = numpy.zeros((num_dofs_per_cell0, num_dofs_per_cell1), dtype=PETSc.ScalarType)
    coordinate_dofs = numpy.zeros([num_dofs_g, gdim])
    orientation = numpy.array([0], dtype=numpy.ubyte)
    coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
    constants = numpy.zeros(1, dtype=PETSc.ScalarType)

    for facet_index in active_facets:
        cell_id = facet_cell[pos_facet[facet_index]]

        # Get local index of facet with respect to the cell
        facets = cell_facet[pos_cell[cell_id]: pos_cell[cell_id + 1]]
        local_facet = numpy.where(facets == facet_index)[0].astype(PETSc.IntType)

        # Get cell coordinates/geometry
        for i in range(num_dofs_g):
            for j in range(gdim):
                coordinate_dofs[i, j] = x_g[cell_g[pos_g[cell_id] + i], j]

        Ae.fill(0.0)
        kernel(ffi.from_buffer(Ae), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants),
               ffi.from_buffer(coordinate_dofs), ffi.from_buffer(local_facet),
               ffi.from_buffer(orientation))

        r = dof_array0[cell_id*num_dofs_per_cell0: cell_id*num_dofs_per_cell0 + num_dofs_per_cell0]
        c = dof_array1[cell_id*num_dofs_per_cell1: cell_id*num_dofs_per_cell1 + num_dofs_per_cell1]
        A.setValuesLocal(r, c, Ae, PETSc.InsertMode.ADD)


B = dolfinx.fem.assemble_matrix(a)
B.assemble()
A.assemble()

print((A-B).norm())
