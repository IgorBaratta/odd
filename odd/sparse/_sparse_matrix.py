import numpy
import scipy.sparse
import scipy.sparse.linalg

from mpi4py import MPI

from odd.core import DistArray
from odd.communication import IndexMap


class DistMatrix(scipy.sparse.linalg.LinearOperator):
    ndim = 2

    def __init__(
        self,
        local_matrix,
        global_shape,
        row_map,
        col_map,
        dtype=float,
        comm=MPI.COMM_WORLD,
    ):
        """
        Distributed Sparse Matrix.

        This can be instantiated in several ways:
            DistMatrix(spmatrix, [shape=(M, N)], rowmap)
                is the standard CSR representation where the column indices for
                with local indices
            DistMatrix(D)
                with a dense distribute 2d array - odd.DistArray
        """
        self.dtype = None
        if isinstance(local_matrix, scipy.sparse.spmatrix):
            self.l_matrix = local_matrix
            self.dtype = local_matrix.dtype
        else:
            raise NotImplementedError

        super().__init__(shape=global_shape, dtype=dtype)

        if not (isinstance(row_map, IndexMap) or isinstance(col_map, IndexMap)):
            raise NotImplementedError

        self.row_map = row_map
        self.col_map = col_map
        self.comm = comm

    def __mul__(self, other):
        if isinstance(other, DistArray):
            return self._matvec(other)
        else:
            return NotImplemented

    def _matvec(self, other: DistArray):
        buffer = numpy.zeros_like(other._array)
        buffer[: self.row_map.local_size] = self.l_matrix @ other._array
        return DistArray(other.shape, buffer=buffer, index_map=other._map)

    def _matmat(self, other):
        return 0

    def get_vector(self):
        return DistArray(shape=self.shape[0], dtype=self.dtype, index_map=self.col_map)

    def matvec(self, x):
        """Matrix-vector multiplication.

        Performs the operation y=A*x where A is a distributed MxN linear
        operator and x is a ditributed array.

        Parameters
        ----------
        x : {DistArray}
            An distributed array with shape (N,) or (N,1).

        Returns
        -------
        y : {DistArray}
            An distributed array with shape (M,) or (M,1) depending
            on the type and shape of the x argument.

        """

        _, N = self.shape

        if x.shape != (N,) and x.shape != (N, 1):
            raise ValueError("dimension mismatch")

        y = self._matvec(x)

        return y

    @property
    def nnz(self):
        sendbuf = numpy.array([self.l_matrix.nnz])
        recvbuf = numpy.zeros_like(sendbuf)
        self.comm.Allreduce(sendbuf, recvbuf)
        return recvbuf[0]

