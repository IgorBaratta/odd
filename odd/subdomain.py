# Copyright (C) 2019 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI
from dolfinx import Function, FunctionSpace, cpp
from .dofmap import DofMap


class SubDomainData():
    def __init__(self,
                 mesh: cpp.mesh.Mesh,
                 V: FunctionSpace,
                 global_comm: MPI.Intracomm):

        # Store dolfinx FunctionSpace object
        self._V = V

        # Store MPI communicator
        self.comm = global_comm

        # Create domain decomposition DofMap
        self.dofmap = DofMap(V)

    @property
    def id(self):
        return self.comm.rank
