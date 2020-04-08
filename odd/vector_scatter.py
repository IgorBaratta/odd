# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from enum import Enum
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import InsertMode, ScatterMode
import numpy
from .index_map import IndexMap

import abc


class ScatterType(Enum):
    """ TODO : class docstring """
    NULL = 0
    PETSc = 1
    RMA = 2
    SHM = 3


class VectorScatter(object, metaclass=abc.ABCMeta):

    _asciiname = "vector scatter "
    _objectname = "vectorscatter"

    """ Manage communication of data between vectors in parallel. """
    def __init__(self, indexmap: IndexMap):
        self.initialized = False
        super(VectorScatter, self).__init__()
        self.indexmap = indexmap

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def reverse(self):
        pass


class PETScVectorScatter(VectorScatter):
    """ TODO : class docstring """
    def __init__(self, indexmap: IndexMap):
        super().__init__(indexmap)

        self._is_local = PETSc.IS().createGeneral(self.indexmap.indices)
        self._vec_scatter = PETSc.Scatter()

    def forward(self, local_vec: PETSc.Vec, global_vec: PETSc.Vec):
        """ TODO : Add descrtiption """
        if not self._vec_scatter.handle:
            self._vec_scatter.create(local_vec, None, global_vec, self._is_local)

        self._vec_scatter(local_vec, global_vec,
                          InsertMode.ADD_VALUES,
                          ScatterMode.SCATTER_FORWARD)

    def reverse(self, local_vec: PETSc.Vec, global_vec: PETSc.Vec):
        """ TODO : Add descrtiption """
        if not self._vec_scatter.handle:
            self._vec_scatter.create(local_vec, None, global_vec, self._is_local)

        self._vec_scatter(global_vec, local_vec,
                          InsertMode.INSERT_VALUES,
                          ScatterMode.SCATTER_REVERSE)


class RMAVectorScatter(VectorScatter):
    """ TODO : class docstring """
    def forward(self, local_vec: PETSc.Vec, global_vec: PETSc.Vec):
        # get data from global to local vector

        # TODO: check the influence of the size of window
        shared_values = local_vec.array[self.dofmap.size_owned:]
        window = MPI.Win.Create(shared_values, comm=self.comm)
        for neighbour in self.dofmap.neighbours:
            buffer_size = 2 * self.dofmap.size_overlap
            buffer = numpy.zeros(buffer_size)
            window.Lock(neighbour, lock_type=MPI.LOCK_SHARED)
            window.Get(buffer, target_rank=neighbour)
            window.Unlock(neighbour)

    def reverse(self, local_vec: PETSc.Vec, global_vec: PETSc.Vec):

        # get data from global to local vector
        window = MPI.Win.Create(global_vec.array, comm=self.comm)

        # Start RMA access epoch
        # window.Start(self.comm.group)

        for neighbour in self.dofmap.neighbours:
            buffer_size = (self.dofmap.all_ranges[neighbour + 1] -
                           self.dofmap.all_ranges[neighbour])
            buffer = numpy.zeros(buffer_size)
            ghosts, indices = self.dofmap.neighbour_ghosts(neighbour)

            # Other processes may access the target window at the same time
            # window.Lock(neighbour, lock_type=MPI.LOCK_SHARED)
            window.Lock_all()
            window.Get(buffer, target_rank=neighbour)
            window.Unlock_all()
            # window.Unlock(neighbour)
            neighbour_offset = self.dofmap.all_ranges[neighbour]
            local_vec.array[indices] = buffer[ghosts - neighbour_offset]
        window.Fence()
        # window.Free()

    def reverse2(self, local_vec: PETSc.Vec, global_vec: PETSc.Vec):
        # local unit size for displacements, in bytes
        disp_unit = global_vec.array.itemsize
        dtype = global_vec.array.dtype
        window = MPI.Win.Create(global_vec.array, disp_unit=disp_unit, comm=self.comm)
        # should be faster in C or C++?
        for i, neighbour in enumerate(self.dofmap.ghost_owners):
            offset = self.dofmap.all_ranges[neighbour]
            local_index = self.dofmap.shared_indices[i] - offset
            buffer = numpy.array(1, dtype=dtype)
            window.Lock(neighbour, lock_type=MPI.LOCK_SHARED)
            window.Get([buffer, MPI.DOUBLE], target_rank=neighbour, target=local_index)
            window.Unlock(neighbour)
            local_vec.array[self.dofmap.size_owned + i] = buffer
        window.Free()
