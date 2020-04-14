# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""
Python wrapper for dolfinx mesh.
We keep only native and numpy objects.
"""

import numpy
import collections
import dolfinx.cpp.mesh


# Data structure to represent Mesh, Geometry and Topology
MeshWrapper = collections.namedtuple("MeshWrapper", "cell topology geometry")
TopologyWrapper = collections.namedtuple("TopologyWrapper", "dim num_cells num_facets boundary_facets")
GeometryWrapper = collections.namedtuple("GeometryWrapper", "dim x x_dofs pos")
CellWrapper = collections.namedtuple("CellWrapper", "num_vertices")


def mesh_wrapper(mesh: dolfinx.cpp.mesh.Mesh):
    mesh.topology.create_connectivity_all()
    topology = topology_wrapper(mesh.topology)
    geometry = geometry_wrapper(mesh.geometry)
    cell = CellWrapper(num_vertices=mesh.ufl_cell().num_vertices())
    return MeshWrapper(cell=cell, topology=topology, geometry=geometry)


def topology_wrapper(topology: dolfinx.cpp.mesh.Topology):
    dim = topology.dim
    num_cells = topology.index_map(dim).size_local + topology.index_map(dim).num_ghosts
    num_facets = topology.index_map(dim - 1).size_local + topology.index_map(dim - 1).num_ghosts
    boundary_facets = numpy.where(topology.on_boundary(dim - 1))[0]
    return TopologyWrapper(dim, num_cells, num_facets, boundary_facets)


def geometry_wrapper(geometry: dolfinx.cpp.mesh.Geometry):
    dim = geometry.dim

    # crop points according to dimension
    x = geometry.x[:, :dim]
    x_dofs = geometry.dofmap.array()
    pos = geometry.dofmap.offsets()
    return GeometryWrapper(dim, x, x_dofs, pos)
