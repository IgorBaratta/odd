# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from ..index_map import IndexMap
from .vector_scatter import VectorScatter, InsertMode
import numpy


class NeighborVectorScatter(VectorScatter):
    def __init__(self, index_map: IndexMap):
        """

        Parameters
        ----------
        index_map : IndexMap
        """
        super().__init__(index_map)
        self.comm = self.index_map.comm

    def forward(self, array: numpy.ndarray, insert_mode: InsertMode = InsertMode.ADD):
        """

        Parameters
        ----------
        array
        insert_mode
        """
        send_indices = self.index_map.ghost_owners.argsort() + self.index_map.owned_size
        send_data = array[send_indices]
        recv_data = numpy.zeros(self.index_map.num_shared_indices).astype(array.dtype)

        self.comm.Neighbor_alltoallv([send_data, (self.index_map.forward_count(), None)],
                                     [recv_data, (self.index_map.reverse_count(), None)])

        # Reduce data before applying to vector
        _, indices = numpy.unique(self.index_map.reverse_indices, return_inverse=True)
        if numpy.iscomplexobj(recv_data):
            reduced_data = numpy.bincount(indices, recv_data.real) + 1j * numpy.bincount(indices, recv_data.imag)
        else:
            reduced_data = numpy.bincount(indices, recv_data)

        array[self.index_map.shared_indices] += reduced_data

    def reverse(self, array: numpy.ndarray, insert_mode: InsertMode = InsertMode.INSERT):
        """

        Parameters
        ----------
        array
        insert_mode
        """
        send_data = array[self.index_map.reverse_indices].astype(array.dtype)
        recv_data = numpy.zeros(self.index_map.ghosts.size).astype(array.dtype)

        self.comm.Neighbor_alltoallv([send_data, (self.index_map.reverse_count(), None)],
                                     [recv_data, (self.index_map.forward_count(), None)])

        if insert_mode == InsertMode.ADD:
            array[self.index_map.owned_size:] += recv_data
        elif insert_mode == InsertMode.INSERT:
            array[self.index_map.owned_size:] = recv_data
