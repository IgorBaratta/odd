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

    The indices are decomposed into three types:
     - Core: Owned indices not shared with other processes
     - Shared: Owned indices shared with neighbor processes
     - Ghosts: Off-processor indices owned by neighbor processes

    Ni = [N_Core + N_Shared] + N_Ghosts

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

        # Define ranges and ghost owners
        recv_buffer = numpy.ndarray(comm.size, dtype=numpy.int32)
        send_buffer = numpy.array(owned_size, dtype=numpy.int32)
        comm.Allgather(send_buffer, recv_buffer)
        all_ranges = numpy.zeros(comm.size + 1, dtype=numpy.int64)
        all_ranges[1:] = numpy.cumsum(recv_buffer)
        self._ghost_owners = numpy.searchsorted(all_ranges, self._ghosts, side="right") - 1
        self._local_range = all_ranges[comm.rank : comm.rank + 2].copy()
        send_neighbors, neighbor_counts = numpy.unique(self._ghost_owners, return_counts=True)

        # "Free" memory - see https://docs.python.org/3/library/gc.html
        del all_ranges

        if comm.rank in self._ghost_owners:
            raise ValueError("Ghost in local range of process " + str(comm.rank))

        send_buffer = numpy.zeros(comm.size, dtype=numpy.int32)
        send_buffer[send_neighbors] = neighbor_counts
        comm.Alltoall(send_buffer, recv_buffer)
        recv_neighbors: ndarray = numpy.flatnonzero(recv_buffer)
        self._num_shared_indices = numpy.sum(recv_buffer)
        self._neighbors = reduce(numpy.union1d, (send_neighbors, recv_neighbors))
        self.comm = comm.Create_dist_graph_adjacent(self._neighbors, self._neighbors)

        self._send_count = numpy.zeros(self._neighbors.size, dtype=numpy.int32)
        send_inds: ndarray = numpy.searchsorted(self._neighbors, send_neighbors, side="right") - 1
        self._send_count[send_inds] = neighbor_counts
        self._recv_count = recv_buffer[self.neighbors]

    @property
    def neighbors(self) -> ndarray:
        """
        Return list of neighbor processes
        """
        return self._neighbors

    @property
    def ghosts(self) -> ndarray:
        """
        Return global indices of ghost entries
        """
        return self._ghosts

    @property
    def local_range(self) -> ndarray:
        """
        Returns the range of indices owned by this processor
        """
        return self._local_range

    @property
    def num_ghosts(self) -> int:
        """
        Returns the range of indices owned by this processor
        """
        return self._ghosts.size

    @property
    def owned_size(self) -> int:
        return self._owned_size

    @property
    def local_size(self) -> int:
        return self.owned_size + self.num_ghosts

    @property
    def num_shared_indices(self) -> int:
        """
        Returns the range of indices owned by this processor
        """
        return self._num_shared_indices

    @property
    def indices(self) -> ndarray:
        """
        Returns global indices including ghosts
        """
        indices = numpy.arange(self.local_size) + self.local_range[0]
        indices[self.owned_size :] = self._ghosts
        return indices

    # noinspection PyAttributeOutsideInit
    @property
    def shared_indices(self) -> ndarray:
        try:
            return self._shared_indices
        except AttributeError:
            # Order ghosts by owner rank
            owners_order = self._ghost_owners.argsort()
            send_data = self._ghosts[owners_order]
            recv_data = numpy.zeros(numpy.sum(self._recv_count), dtype=numpy.int64)
            self.comm.Neighbor_alltoallv([send_data, (self._send_count, None)], [recv_data, (self._recv_count, None)])
            self._shared_indices = recv_data
            return recv_data
