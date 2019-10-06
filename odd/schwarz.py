from enum import Enum
import numpy
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import IntType
from .subdomain import SubDomainData
from .dofmap import DofMap
from .vector_scatter import PETScVectorScatter


class SMType(Enum):
    """ TODO : class docstring """
    restricted = 1
    additive = 2
    multiplicative = 3


class AdditiveSchwarz():
    """ TODO : class docstring """
    def __init__(self, data: SubDomainData, A: PETSc.Mat):

        self.dofmap = data.dofmap
        self.comm = data.comm

        # Restricted Additive Schwarz is the default type
        self._type = SMType.restricted

        # set preconditioner state
        self.state = False

        if (A.type == PETSc.Mat.Type.MPIAIJ):
            # A is an assembled distributed global matrix
            self._is_local = PETSc.IS().createGeneral(self.dofmap.indices)
            self.Ai = A.createSubMatrices(self._is_local)[0]
            self.vec_global = A.getVecRight()
        elif (A.type == PETSc.Mat.Type.SEQAIJ):
            # A is a sequential local matrix for each process
            self.vec_global = data.global_vec()
            self.Ai = A
        else:
            raise Exception('Wrong matrix type')

        # Declare some variables, but leave the initialization
        # for setup step
        self.Ri = PETSc.Mat().create(MPI.COMM_SELF)
        self.Di = PETSc.Mat().create(MPI.COMM_SELF)
        self.solver = PETSc.KSP().create(MPI.COMM_SELF)

    def setUp(self, pc=None):

        # Create local working vectors, sequential
        # compatible with the local sequential matrix
        self.vec1, self.vec2 = self.Ai.getVecs()

        # Assemble restrictiton and partition of unity matrices
        if not self.Ri.isAssembled():
            self.Ri = self.restritction_matrix(self.dofmap)
        if not self.Di.isAssembled():
            self.Di = self.partition_of_unity(self.dofmap, "owned")

        # Check if there is a valid local solver and create
        # default it doesn't exist
        if not self.solver.type or self.solver.comm != MPI.COMM_SELF:
            self.solver = PETSc.KSP().create(MPI.COMM_SELF)
            self.solver.setOperators(self.Ai)
            self.solver.setType('preonly')
            self.solver.pc.setType('lu')
            self.solver.pc.setFactorSolverType('mumps')
            self.solver.setFromOptions()

        self.scatter = PETScVectorScatter(self.comm, self.dofmap,
                                          self.vec1, self.vec_global)

        # Define state of preconditioner, True means ready to use
        self.state = True

    def global_matrix(self) -> PETSc.Mat:
        '''
        Return the unassembled global matrix of type python.
        '''
        if self.state:
            # Define global unassembled global Matrix A
            self.A = PETSc.Mat().create()
            self.A.setSizes(self.sizes)
            self.A.setType(self.A.Type.PYTHON)
            self.A.setPythonContext(self)
            self.A.setUp()
            return self.A
        else:
            raise Exception('Matrix in wrong state.'
                            'Please setUP preconditioner first.')

    def global_vector(self, bi: PETSc.Vec, destroy=False) -> PETSc.Vec:
        # TODO: Avoid making unecessary data duplication
        # this may not be working very well....need some testing
        if self.state:
            if bi.comm.size == 1:
                b = self.vec_global.duplicate()
                self.vector_scatter(bi, b,
                                    PETSc.InsertMode.INSERT_VALUES,
                                    PETSc.ScatterMode.SCATTER_FORWARD)
                return b
            else:
                raise Exception('Should be a sequential vector')
        else:
            raise Exception('Preconditioner in wrong state.'
                            'Please setUP preconditioner first.')

    @property
    def type(self):
        return self._type

    @property
    def sizes(self):
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
        # have to zero vector, may contain garbage
        b.zeroEntries()

        self.scatter.reverse(self.vec1, x)

        self.solver.solve(self.vec1, self.vec2)

        if self.type == SMType.restricted:
            self.Di.mult(self.vec2, self.vec1)
        elif self.type == SMType.additive:
            self.vec1.array = self.vec2.array
        else:
            raise RuntimeError("Not implemented")

        self.scatter.forward(self.vec1, b)

    def mult(self, mat: PETSc.Mat, x: PETSc.Vec, b: PETSc.Vec):
        # have to zero vector, may contain garbage
        b.zeroEntries()

        self.scatter.reverse(self.vec1, x)

        self.Ai.mult(self.vec1, self.vec2)
        self.Di.mult(self.vec2, self.vec1)

        self.scatter.forward(self.vec1, b)

    @staticmethod
    def restritction_matrix(dofmap: DofMap) -> PETSc.Mat:
        """
        Explicitely construct the local restriction matrix for
        the current subdomain.''
        """

        # number of non-zeros per row
        nnz = 1

        # Local Size, including overlap
        N = dofmap.size_local

        # Global Size
        N_global = dofmap.size_global

        # create restriction data in csr format
        A = numpy.ones(N, dtype=IntType)
        IA = numpy.arange(N + 1, dtype=IntType)
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

    @staticmethod
    def partition_of_unity(dofmap: DofMap, mode="owned") -> PETSc.Mat:
        """
        Return the assembled partition of unit matrix for the current
        subdomain/process.
        """

        # number of non-zeros per row
        nnz = 1
        N = dofmap.size_local
        N_owned = dofmap.size_owned

        # create restriction data in csr format
        A = numpy.zeros(N, dtype=IntType)
        A[0:N_owned] = 1
        IA = numpy.arange(N + 1, dtype=IntType)
        JA = numpy.arange(N, dtype=IntType)

        # Create and assemble Partition of Unity Matrix
        D = PETSc.Mat().create(MPI.COMM_SELF)
        D.setType('aij')
        D.setSizes([N, N])
        D.setPreallocationNNZ(nnz)
        D.setValuesCSR(IA, JA, A)
        D.assemblyBegin()
        D.assemblyEnd()

        return D
