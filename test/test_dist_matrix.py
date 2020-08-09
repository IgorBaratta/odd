from scipy import sparse
from mpi4py import MPI

import pytest
import numpy
import odd.sparse


def eye(comm, N):
    if comm.rank == 0:
        diags = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        mat = sparse.eye(N, k=0).tocsr()
        for k in diags:
            mat += sparse.eye(N, k=k)
    else:
        mat = None
    return mat


@pytest.mark.parametrize("N", ["1000", "10000"])
def test_dist_mat_vec(N):
    comm = MPI.COMM_WORLD
    mat = eye(comm, N)
    A = odd.sparse.distribute_csr_matrix(mat, comm)

    b = A.get_vector()
    b.fill(comm.rank)

    # before ghost update
    assert numpy.all(b.ghost_values == comm.rank)
    b.update()

    # after updating ghost values
    assert numpy.all(b.ghost_values == b._map.ghost_owners)

    x = A.matvec(b)
    my_array = comm.gather(x.array)
    b_array = comm.gather(b.array)

    if comm.rank == 0:
        arr = numpy.hstack(my_array)
        b_0 = numpy.hstack(b_array)
        x_0 = mat @ b_0
        assert numpy.allclose(arr, x_0)


@pytest.mark.parametrize("mat_str", ["FEMLAB/poisson2D", "GHS_indef/helm3d01"])
def test_matvec_suitesparse(mat_str):
    comm = MPI.COMM_WORLD
    mat = odd.sparse.get_csr_matrix(mat_str, False, comm=MPI.COMM_WORLD)
    A = odd.sparse.distribute_csr_matrix(mat, comm)

    b = A.get_vector()
    b.fill(1)

    x = A.matvec(b)
    my_array = comm.gather(x.array)
    b_array = comm.gather(b.array)

    if comm.rank == 0:
        arr = numpy.hstack(my_array)
        b_0 = numpy.hstack(b_array)
        x_0 = mat @ b_0
        assert numpy.allclose(arr, x_0)
