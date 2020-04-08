# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
from mpi4py import MPI
from functools import reduce
from typing import List, Union

from numpy.core.multiarray import ndarray


class IndexMap(object):
    """
    This class manages the map between the local indices (sub-domain level) and the global parallel ordering,
    accounting for ghost padding.

    The indices are decomposed into three types, and reordered locally:
     - Core: Owned indices not shared with other processes
     - Shared: Owned indices shared with neighbor processes
     - Ghosts: Off-processor indices owned by neighbor processes

    Ni = N_Core + N_Shared + N_Ghosts

    Notes
    -----
    This class is a simplified version  of dolfinx IndexMap.
    """
    def __init__(self, comm: MPI.Intracomm, owned_size: int, ghosts: Union[List, ndarray]):
        """
        Parameters
        ----------
        comm:
            The MPI communicator to use
        owned_size:
            The number of owned degrees of freedom (Core + Shared)
        ghosts:
            The global indices of ghost entries, or empty array if not needed.
        """
        self._ghosts = numpy.array(ghosts, dtype=numpy.int64)
        self._owned_size = owned_size
        recv_buffer = numpy.ndarray(comm.size, dtype=numpy.int32)
        send_buffer = numpy.array(owned_size, dtype=numpy.int32)
        comm.Allgatherv(send_buffer, recv_buffer)
        self._all_ranges = numpy.zeros(comm.size + 1, dtype=numpy.int64)
        self._all_ranges[1:] = numpy.cumsum(recv_buffer)
        self._ghost_owners = numpy.searchsorted(self._all_ranges, self._ghosts, side='right') - 1
        send_neighbors, send_counts = numpy.unique(self._ghost_owners, return_counts=True)

        if comm.rank in self._ghost_owners:
            raise ValueError('Ghost in local range of process ' + str(comm.rank))

        if isinstance(comm, MPI.Distgraphcomm):
            # Check if comm is already a distributed graph topology communicator
            # and get neighboring processes
            self.comm = comm
            self._neighbors = reduce(numpy.union1d, comm.inoutedges)
        else:
            # Check if it is already a topology communicator and get neighbor processes
            send_buffer = numpy.zeros(comm.size, dtype=numpy.int32)
            send_buffer[send_neighbors] = send_counts
            comm.Alltoall(send_buffer, recv_buffer)
            recv_neighbors = numpy.flatnonzero(recv_buffer)
            # noinspection PyTypeChecker
            self._num_shared_indices = numpy.sum(recv_neighbors, axis=0)
            self._neighbors = reduce(numpy.union1d, (send_neighbors, recv_neighbors))
            self.comm = comm.Create_dist_graph_adjacent(self._neighbors, self._neighbors)

        # owners_ordered = self._ghost_owners.argsort()
        # send_data = ghosts[owners_ordered].copy()
        # send_count = numpy.zeros(self._neighbors.size, dtype=numpy.int32)
        # inds: Union[ndarray, int] = numpy.searchsorted(self.neighbors, send_neighbors, side='right') - 1
        # send_count[inds] = send_counts
        #
        # recv_count = recv_buffer[self.neighbors]
        # recv_data = numpy.zeros(numpy.sum(recv_count), dtype=numpy.int64)
        # self.comm.Neighbor_alltoallv([send_data, (send_count, None)],
        #                              [recv_data, (recv_count, None)])

    @property
    def neighbors(self) -> numpy.ndarray:
        """
        Return list of neighbor processes
        """
        return self._neighbors

    @property
    def local_range(self) -> numpy.ndarray:
        """
        Returns the range of indices owned by this processor
        """
        rank = self.comm.rank
        return self._all_ranges[rank:rank+2].copy()

    @property
    def num_ghosts(self) -> int:
        """
        Returns the range of indices owned by this processor
        """
        return self._ghosts.size

    @property
    def owned_size(self) -> int:
        return self._owned_size + self.num_ghosts

    @property
    def local_size(self) -> int:
        return self._owned_size + self.num_ghosts

    @property
    def num_shared_indices(self) -> int:
        """
        Returns the range of indices owned by this processor
        """
        return self._num_shared_indices.size

    @property
    def shared_indices(self) -> numpy.ndarray:
        return self._neighbors

    @property
    def indices(self) -> numpy.ndarray:
        indices = numpy.arange(self.local_size)
        indices[self.owned_size:] = self._ghosts
        return indices






















