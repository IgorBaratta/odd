from enum import Enum, unique

from mpi4py import MPI
from numpy import arange, ones, zeros

from petsc4py import PETSc
from petsc4py.PETSc import IntType
from SubDomain import SubDomainData


class AdditiveSchwarz():
    def __init__(self, data: SubDomainData, A: PETSc.Mat):
        self.dofmap = data.dofmap

        self.Ri = restritction_matrix(self.dofmap)
        self.Di = partition_of_unity(self.dofmap)

        self.is_owned = PETSc.IS().createGeneral(self.dofmap.owned_indices)
        self.is_local = PETSc.IS().createGeneral(self.dofmap.indices)

        if (A.comm.size == data.comm.size):
            self.Ai = A.createSubMatrices(self.is_local)[0]
            self.vglobal = A.getVecRight()
        elif (A.comm.size == 1):
            self.Ai = A
        else:
            raise Exception('Wrong matrix type')

        self.vec1, self.vec2 = self.Ai.getVecs()

        self.solver = PETSc.KSP().create(MPI.COMM_SELF)
        self.solver.setOperators(self.Ai)
        self.solver.setType('preonly')
        self.solver.pc.setType('lu')
        self.solver.pc.setFactorSolverType('mumps')
        self.solver.setFromOptions()

        self._type = ASMType.restricted

        self.vector_scatter = PETSc.Scatter().create(self.vec1,
                                                     None,
                                                     self.vglobal,
                                                     self.is_local)

    @property
    def type(self):
        return self._type

    def PETScSizes(self):
        N_owned = self.dofmap.size_owned
        Ng = self.dofmap.size_global
        return ((N_owned, Ng), (N_owned, Ng))

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, b: PETSc.Vec):
        """
        Pure PETSc implementation of the restricted additve schwarz

        Parameters
        ==========

        pc: This argument is not called within the function but it
            belongs to the standard way of calling a preconditioner.

        x : petsc.Vec
            The vector to which the preconditioner is to be applied.

        b : petsc.Vec
            The vector that stores the result of the preconditioning
            operation.

        """
        # May contain garbage
        b.zeroEntries()

        self.vector_scatter(x, self.vec1,
                            PETSc.InsertMode.INSERT_VALUES,
                            PETSc.ScatterMode.SCATTER_REVERSE)

        self.solver.solve(self.vec1, self.vec2)

        if self.type == ASMType.restricted:
            self.Di.mult(self.vec2, self.vec1)
        elif self.type == ASMType.additive:
            self.vec1.array = self.vec2.array
        else:
            raise RuntimeError("Not implemented")

        self.vector_scatter(self.vec1, b,
                            PETSc.InsertMode.ADD_VALUES,
                            PETSc.ScatterMode.SCATTER_FORWARD)


class ASMType(Enum):
    restricted = 1
    additive = 2
    multiplicative = 3


def restritction_matrix(dofmap):
    """
    Explicitely construct the local restriction matrix for
    the current subdomain.''
    """

    # number of non-zeros per row
    nnz = 1

    # Local Size
    N = dofmap.size_local

    # Global Size
    N_global = dofmap.size_global

    # create restriction data in csr format
    A = ones(N, dtype=IntType)
    IA = arange(N + 1, dtype=IntType)
    JA = dofmap.indices

    # Create and assembly local Restriction Matrix
    R = PETSc.Mat().create(MPI.COMM_SELF)
    R.setType('aij')
    R.setSizes([N, N_global])
    R.setPreallocationNNZ(nnz)
    R.setValuesCSR(IA, JA, A)
    R.assemblyBegin()
    R.assemblyEnd()

    return R


def partition_of_unity(dofmap, mode="owned"):
    """
    Create partition of unit matrix for
    for the current subdomain.
    """

    # create restriction data in csr format
    nnz = 1  # number of non-zeros per row
    N = dofmap.size_local
    N_owned = dofmap.size_owned

    A = zeros(N, dtype=IntType)
    A[0:N_owned] = 1
    IA = arange(N + 1, dtype=IntType)
    JA = arange(N, dtype=IntType)

    # Create and assembly Restriction Matrix
    D = PETSc.Mat().create(MPI.COMM_SELF)
    D.setType('aij')
    D.setSizes([N, N])
    D.setPreallocationNNZ(nnz)
    D.setValuesCSR(IA, JA, A)
    D.assemblyBegin()
    D.assemblyEnd()

    return D
