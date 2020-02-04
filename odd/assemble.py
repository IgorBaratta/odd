# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from petsc4py import PETSc
import numpy
import dolfinx
import ufl


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


def assemble_vector(L: ufl.Form)->PETSc.Vec:
    """
    Communicationless assemble provided FFCX/UFC kernel over a mesh into the array b
    """
    _L = _create_cpp_form(L)
    if _L.rank != 1:
        raise ValueError
    mesh = _L.mesh()

    b = dolfinx.fem.create_vector(_L)

    # Define active cells
    dim = mesh.topology.dim
    num_cells = mesh.topology.size(dim)

    # Define active cells
    active_cells = numpy.arange(num_cells)
    int_cell = dolfinx.cpp.fem.FormIntegrals.Type.cell

    # Define active facets
    on_boundary = mesh.topology.on_boundary
    active_facets = numpy.where(on_boundary)[0]
    int_facet = dolfinx.cpp.fem.FormIntegrals.Type.exterior_facet

    _L.set_active_entities(int_cell, 0, active_cells)
    # _L.set_active_entities(int_facet, 0, active_facets)
    dolfinx.cpp.fem.assemble_vector(b, _L)

    return b
