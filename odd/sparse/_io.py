import tarfile
import urllib.request
import os
from scipy.io import mmread
from scipy.sparse import csr_matrix
from mpi4py import MPI

from odd.utils import partition1d
from odd.communication import IndexMap
from ._sparse_matrix import DistMatrix

import numpy


def get_csr_matrix(Name: str, verbose: bool = True, comm=MPI.COMM_WORLD) -> csr_matrix:
    """
    Get matrix from the SuiteSparse Matrix Collection website and
    convert to the a distributed sparse matrix in the odd.sparse_matrix format.

    The method only open the files or read from input streams on MPI Process 0,
    and redistribute data to the other MPI processes. It can be quite expensive.

    """
    base_url = "https://suitesparse-collection-website.herokuapp.com/MM/"
    url = base_url + Name + ".tar.gz"
    infile = Name.split("/")[1]
    dest_file = infile + "/" + infile + ".mtx"

    if comm.rank == 0:

        # Download the file only if it does not exist
        if os.path.isfile(dest_file):
            if verbose:
                print("\t -----------------------------------------------------------")
                print("\t File already exists.")
        else:
            if verbose:
                print("\t -----------------------------------------------------------")
                print("\t Downloading matrix file from suitesparse collection")
            urllib.request.urlretrieve(url, infile + ".tar.gz")

            if verbose:
                print("\t -----------------------------------------------------------")
                print("\t Extrating tar.gz file to folder ./", infile)
            tar = tarfile.open(infile + ".tar.gz")
            tar.extractall()
            tar.close()

        if verbose:
            print("\t -----------------------------------------------------------")
            print("\t Reading matrix and converting to csr format")
        A = mmread(dest_file)
        A = A.tocsr()

        if verbose:
            print("\t -----------------------------------------------------------")
            print("\t Done! \n")

    # Distribute matrix coo matrix
    if comm.rank != 0:
        A = None

    return A


def distribute_csr_matrix(A, comm=MPI.COMM_WORLD, root=0) -> DistMatrix:
    myrank = comm.rank
    shape = A.shape if myrank == root else None

    # Broadcast some matrix data
    shape = comm.bcast(shape, root)
    rowmap = partition1d(comm, shape[0], overlap=0)

    # Distribute indices and data
    l_indptr = distribute_indptr(rowmap, A, root=0)
    l_data = distribute_data(rowmap, A, l_indptr, root=0)
    indices = distribute_indices(rowmap, A, l_indptr, root=0)

    left, right = rowmap.local_range
    ghosts = indices[numpy.logical_or(indices < left, indices >= right)]
    ghosts = numpy.unique(ghosts)

    colmap = IndexMap(comm, rowmap.owned_size, ghosts)
    l_indices = colmap.global_to_local(indices)
    l_indptr = l_indptr - l_indptr[0]
    l_matrix = csr_matrix(
        (l_data, l_indices, l_indptr), shape=[rowmap.local_size, colmap.local_size]
    )

    return DistMatrix(l_matrix, shape, row_map=rowmap, col_map=colmap)


def distribute_indptr(rowmap, matrix, root=0) -> numpy.ndarray:
    """
    Collective call
    """

    comm = MPI.COMM_WORLD
    sendbuf = numpy.array([rowmap.owned_size])
    if comm.rank == root:
        recvbuf = numpy.empty(comm.size, dtype=sendbuf.dtype)
    else:
        recvbuf = None

    comm.Gather(sendbuf, recvbuf, root)
    all_sizes = numpy.copy(recvbuf)

    displ = numpy.cumsum(all_sizes)
    displ = numpy.insert(displ, 0, 0)[:-1]
    count = all_sizes + 1 if comm.rank == root else all_sizes
    mpi_type = MPI._typedict["l"]

    sendbuf = None
    if comm.rank == root:
        sendbuf = matrix.indptr.astype(numpy.int64)

    recv_buffer = numpy.empty(rowmap.owned_size + 1, dtype=numpy.int64)
    comm.Scatterv([sendbuf, count, displ, mpi_type], recv_buffer, root)
    indptr = numpy.copy(recv_buffer)

    return indptr


def distribute_data(rowmap, A, l_indptr, root=0) -> numpy.ndarray:
    comm = MPI.COMM_WORLD
    myrank = comm.rank

    dtype = A.dtype if myrank == root else None
    dtype = comm.bcast(dtype, root)

    l_nnz = l_indptr[-1] - l_indptr[0]

    sendbuf = numpy.array([l_nnz])
    recvbuf = numpy.empty(comm.size, dtype=sendbuf.dtype) if myrank == root else None
    comm.Gather(sendbuf, recvbuf, root)

    count = numpy.copy(recvbuf)
    displ = numpy.cumsum(count)
    displ = numpy.insert(displ, 0, 0)[:-1]
    mpi_type = MPI._typedict[dtype.char]

    sendbuf = A.data if myrank == root else None
    recv_buffer = numpy.empty(l_nnz, dtype=dtype)
    comm.Scatterv([sendbuf, count, displ, mpi_type], recv_buffer, root)

    return recv_buffer


def distribute_indices(rowmap, A, l_indptr, root=0) -> numpy.ndarray:
    comm = MPI.COMM_WORLD
    myrank = comm.rank

    l_nnz = l_indptr[-1] - l_indptr[0]
    sendbuf = numpy.array([l_nnz])
    recvbuf = numpy.empty(comm.size, dtype=sendbuf.dtype) if myrank == root else None
    comm.Gather(sendbuf, recvbuf, root)

    count = numpy.copy(recvbuf)
    displ = numpy.cumsum(count)
    displ = numpy.insert(displ, 0, 0)[:-1]
    mpi_type = MPI._typedict["l"]

    sendbuf = A.indices.astype("l") if myrank == root else None
    recv_buffer = numpy.empty(l_nnz, dtype="l")
    comm.Scatterv([sendbuf, count, displ, mpi_type], recv_buffer, root)

    return recv_buffer
