from mpi4py import MPI
import numpy


MPI_OP = {"amax": MPI.MAX, "amin": MPI.MIN, "sum": MPI.SUM}


def mpi_reduction(dist_array, func, **kwargs):
    comm: MPI.Intracomm = dist_array.mpi_comm
    local = func(dist_array.array, **kwargs)
    sendbuf = numpy.asarray(local)

    if (func.__name__ in MPI_OP) and not numpy.iscomplexobj(dist_array):
        recvbuf = numpy.zeros(sendbuf.size, dtype=sendbuf.dtype)
        comm.Allreduce(sendbuf, recvbuf, MPI_OP[func.__name__])
        return recvbuf
    else:
        recvbuf = numpy.zeros(comm.size * sendbuf.size, dtype=sendbuf.dtype)
        comm.Allgather(sendbuf, recvbuf)
        return func(recvbuf, **kwargs)


def dot1d(a, b):
    ldot = numpy.dot(a.array, b.array)
    sendbuf = numpy.asarray(ldot)
    recvbuf = numpy.zeros_like(sendbuf)
    comm: MPI.Intracomm = a.mpi_comm
    assert a.mpi_comm.size == b.mpi_comm.size
    comm.Allreduce(sendbuf, recvbuf, MPI.SUM)
    return recvbuf


def vdot1d(a, b):
    ldot = numpy.vdot(a.array, b.array)
    sendbuf = numpy.asarray(ldot)
    recvbuf = numpy.zeros_like(sendbuf)
    comm: MPI.Intracomm = a.mpi_comm
    assert a.mpi_comm.size == b.mpi_comm.size
    comm.Allreduce(sendbuf, recvbuf, MPI.SUM)
    return recvbuf
