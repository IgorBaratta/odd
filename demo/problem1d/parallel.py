from mpi4py import MPI
import numpy

# Problem data
comm = MPI.COMM_WORLD
owned_size = 10
num_ghosts = 2

neighbor = (comm.rank + 1) % comm.size
ghosts = neighbor * owned_size + numpy.arange(num_ghosts)
ghost_owners = None

if ghost_owners is None:
    # Define ranges and ghost owners
    # Todo: Check if its really necessary, if we already know the global index of ghosts,
    #       wouldn't we have to know the global numbering in that rank?
    recv_buffer = numpy.ndarray(comm.size, dtype=numpy.int32)
    send_buffer = numpy.array(owned_size, dtype=numpy.int32)

    # Allgather is expected to be O(p)
    comm.Allgather(send_buffer, recv_buffer)

    all_ranges = numpy.zeros(comm.size + 1, dtype=numpy.int64)
    all_ranges[1:] = numpy.cumsum(recv_buffer)
    ghost_owners = numpy.searchsorted(all_ranges, ghosts, side="right") - 1
    del all_ranges  # "Free" memory - see https://docs.python.org/3/library/gc.html

# The ghosts in the current process are owned by reverse_neighbors
# Reverse counts is the number of ghost indices grouped by ghost owner.
reverse_neighbors, reverse_counts = numpy.unique(ghost_owners, return_counts=True)

# Define how much data to send to each process in reverse mode
send_buffer = numpy.zeros(comm.size, dtype=numpy.int32)
send_buffer[reverse_neighbors] = reverse_counts

# Define how much data to receive from each process in reverse mode
recv_buffer = numpy.ndarray(comm.size, dtype=numpy.int32)

# Alltoall is expected to be O(p log(p))
comm.Alltoall(send_buffer, recv_buffer)

# The current process owns indices that are ghosts in forward_neighbors
forward_neighbors = numpy.flatnonzero(recv_buffer)
forward_counts = recv_buffer[forward_neighbors]


# sources -	ranks of processes for which the calling process is a destination
# destinations - ranks of processes for which the calling process is a destination
reverse_comm = comm.Create_dist_graph_adjacent(sources=forward_neighbors,
                                               destinations=reverse_neighbors)
forward_comm = comm.Create_dist_graph_adjacent(sources=reverse_neighbors,
                                               destinations=forward_neighbors)

# Order ghosts by owner rank
owners_order = ghost_owners.argsort()
send_data = ghosts[owners_order]
recv_data = numpy.zeros(numpy.sum(forward_counts), dtype=numpy.int64)
reverse_comm.Neighbor_alltoallv([send_data, (reverse_counts, None)],
                                [recv_data, (forward_counts, None)])


recv_buffer = numpy.array([0], dtype=numpy.int32)
send_buffer = numpy.array([owned_size], dtype=numpy.int32)
comm.Exscan(send_buffer, recv_buffer)
print(recv_buffer)