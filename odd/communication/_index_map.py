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
from ._utils import global_to_local_numba


class IndexMap(object):
    """
    This class manages the map between the local indices (process level) and the global parallel ordering,
    accounting for ghost padding.

    The indices are decomposed into three types:
     - Core: Owned indices not shared with other processes
     - Shared: Owned indices shared with neighbor processes
     - Ghosts: Off-processor indices owned by neighbor processes

    Ni = [N_Core + N_Shared] + N_Ghosts

    Notes
    -----
    This class is a simplified version  of dolfinx IndexMap.
    Todo: Allow user to set integer types.
          Local indices are 32 bit integers and global indices are 64 bit integers.
    """

    def __init__(
        self,
        comm: MPI.Intracomm,
        owned_size: int,
        ghosts: Union[List, ndarray],
        ghost_owners: Union[List, ndarray] = None,
    ):
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
        self._ghost_owners = ghost_owners

        self._local_range = numpy.zeros(2, dtype=numpy.int64)
        if self._ghost_owners is None:
            # Define ranges and ghost owners
            recv_buffer = numpy.ndarray(comm.size, dtype=numpy.int32)
            send_buffer = numpy.array(owned_size, dtype=numpy.int32)

            # Allgather is expected to be O(p)
            comm.Allgather(send_buffer, recv_buffer)

            all_ranges = numpy.zeros(comm.size + 1, dtype=numpy.int64)
            all_ranges[1:] = numpy.cumsum(recv_buffer)
            self._ghost_owners = (
                numpy.searchsorted(all_ranges, self._ghosts, side="right") - 1
            )
            self._local_range[0] = all_ranges[comm.rank]
            self._local_range[1] = all_ranges[comm.rank + 1]
            # The memory of all_ranges is collected by the garbage collector,
            # there is no need to free memory, see https://docs.python.org/3/library/gc.html
        else:
            send_buffer = numpy.array([owned_size], dtype=numpy.int32)
            comm.Exscan(send_buffer, self._local_range[:-1])
            self._local_range[1] = self._local_range[0] + owned_size

        if comm.rank in self._ghost_owners:
            raise ValueError("Ghost in local range of process " + str(comm.rank))

        # The ghosts in the current process are owned by reverse_neighbors
        # Reverse counts is the number of ghost indices grouped by ghost owner.
        self.reverse_neighbors, self.reverse_counts = numpy.unique(
            self._ghost_owners, return_counts=True
        )
        send_buffer = numpy.zeros(comm.size, dtype=numpy.int32)
        send_buffer[self.reverse_neighbors] = self.reverse_counts

        # Define how much data to receive from each process in reverse mode
        recv_buffer = numpy.ndarray(comm.size, dtype=numpy.int32)
        # Alltoall is expected to be O(p log(p))
        comm.Alltoall(send_buffer, recv_buffer)
        # The current process owns indices that are ghosts in forward_neighbors
        self.forward_neighbors = numpy.flatnonzero(recv_buffer)
        self.forward_counts = recv_buffer[self.forward_neighbors]

        # Create a communicator for both forward and reverse modes.
        # sources -	ranks of processes for which the calling process is a destination
        # destinations - ranks of processes for which the calling process is a destination
        self.reverse_comm = comm.Create_dist_graph_adjacent(
            sources=self.forward_neighbors, destinations=self.reverse_neighbors
        )
        self.forward_comm = comm.Create_dist_graph_adjacent(
            sources=self.reverse_neighbors, destinations=self.forward_neighbors
        )

    @property
    def neighbors(self) -> ndarray:
        """
        Return list of neighbor processes
        """
        return reduce(numpy.union1d, (self.reverse_neighbors, self.forward_neighbors))

    @property
    def global_size(self) -> numpy.int64:
        return self.forward_comm.allreduce(self.owned_size)

    @property
    def ghost_owners(self) -> ndarray:
        """
        Return list of neighbor processes
        """
        return self._ghost_owners

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
    def shift(self):
        return self.local_range[0]

    @property
    def indices(self) -> ndarray:
        """
        Returns global indices including ghosts
        """
        indices = numpy.arange(self.local_size) + self.local_range[0]
        indices[self.owned_size :] = self._ghosts
        return indices

    def ordered_indices(self):
        return numpy.sort(self.indices)

    @property
    def reverse_indices(self) -> ndarray:
        # Order ghosts by owner rank
        owners_order = self._ghost_owners.argsort()
        send_data = self._ghosts[owners_order]
        recv_data = numpy.zeros(numpy.sum(self.forward_counts), dtype=numpy.int64)

        # Send reverse_counts ghost indices to reverse neighbors and receive forward_counts
        # owned indices from forward neighbors
        self.reverse_comm.Neighbor_alltoallv(
            [send_data, (self.reverse_counts, None)],
            [recv_data, (self.forward_counts, None)],
        )
        return recv_data - self.shift

    @property
    def shared_indices(self) -> ndarray:
        return numpy.unique(self.reverse_indices)

    @property
    def num_shared_indices(self) -> int:
        """
        Returns the range of indices owned by this processor
        """
        return self.shared_indices.size

    def reverse_count(self, block_size=1):
        return self.reverse_counts * block_size

    def forward_count(self, block_size=1):
        return self.forward_counts * block_size

    def global_to_local(self, indices):
        return global_to_local_numba(indices, self.local_range, self.ghosts)
