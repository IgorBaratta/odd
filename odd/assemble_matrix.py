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


mesh = dolfinx.UnitSquareMesh(dolfinx.MPI.comm_self, 100, 100)
V = dolfinx.FunctionSpace(mesh, ("Lagrange", 3))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + ufl.inner(u, v)*ufl.dx


# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'), numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'), numba.types.complex64)


def assemble_matrix(a):
    _a = _create_cpp_form(a)
    A = dolfinx.cpp.fem.create_matrix(_a)
    ufc_form = dolfinx.jit.ffcx_jit(a)
    mesh = _a.mesh()
    dim = mesh.topology.dim

    dofmap0 = _a.function_space(0).dofmap
    dofmap1 = _a.function_space(1).dofmap
    dof_array0 = dofmap0.dof_array()
    dof_array1 = dofmap1.dof_array()

    num_dofs_per_cell0 = dofmap0.dof_layout.num_dofs
    num_dofs_per_cell1 = dofmap1.dof_layout.num_dofs

    if ufc_form.num_cell_integrals == 1:
        num_cells = mesh.num_entities(dim)
        active_cells = numpy.arange(num_cells)

        cell_g = mesh.topology.connectivity(dim, 0).connections()
        pos_g = mesh.topology.connectivity(dim, 0).pos()
        x_g = mesh.geometry.points
        num_dofs_g = mesh.topology.connectivity(dim, 0).size(0)

        # Pack data to call function
        mesh_data = (dim, x_g, cell_g, pos_g)
        num_dofs = (num_dofs_per_cell0, num_dofs_per_cell1, num_dofs_g)
        dofs = (dof_array0, dof_array1)

        kernel = ufc_form.create_cell_integral(-1).tabulate_tensor
        assemble_cells(A, mesh_data, dofs, num_dofs, active_cells, kernel)

    return A


def assemble_cells(A, mesh_data, dofs, num_dofs, active_cells, kernel):
    num_dofs_per_cell0, num_dofs_per_cell1, num_dofs_g = num_dofs
    dof_array0, dof_array1 = dofs
    gdim, x_g, cell_g, pos_g = mesh_data

    Ae = numpy.zeros((num_dofs_per_cell0, num_dofs_per_cell1), dtype=PETSc.ScalarType)
    coordinate_dofs = numpy.zeros([num_dofs_g, gdim])
    orientation = numpy.array([0], dtype=numpy.int32)
    coeffs = numpy.zeros(1, dtype=PETSc.ScalarType)
    constants = numpy.zeros(1, dtype=PETSc.ScalarType)
    coordinate_dofs = numpy.zeros([num_dofs_g, gdim])

    for idx in active_cells:
        # Get cell coordinates/geometry
        for i in range(num_dofs_g):
            for j in range(gdim):
                coordinate_dofs[i, j] = x_g[cell_g[pos_g[idx] + i], j]

        Ae.fill(0.0)
        kernel(ffi.from_buffer(Ae), ffi.from_buffer(coeffs),
               ffi.from_buffer(constants),
               ffi.from_buffer(coordinate_dofs), ffi.from_buffer(orientation),
               ffi.from_buffer(orientation))

        rows = dof_array0[idx*num_dofs_per_cell0: idx*num_dofs_per_cell0 + num_dofs_per_cell0]
        cols = dof_array1[idx*num_dofs_per_cell1: idx*num_dofs_per_cell1 + num_dofs_per_cell1]
        A.setValuesLocal(rows, cols, Ae, PETSc.InsertMode.ADD)


t1 = Timer("xxxxx - odd")
A = assemble_matrix(a)
A.assemble()
t1.stop()

t2 = Timer("xxxxx - dolfinx")
B = dolfinx.fem.assemble_matrix(a)
B.assemble()
t2.stop()

print((A-B).norm())

list_timings(dolfinx.MPI.comm_world, [TimingType.wall])
