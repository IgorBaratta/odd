# Copyright (C) 2019 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from enum import Enum
import numpy
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import IntType
from .subdomain import SubDomainData
from .dofmap import DofMap
from .vector_scatter import PETScVectorScatter, RMAVectorScatter, ScatterType


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
        self.pc_type = SMType.additive

        # set preconditioner state
        self.state = False

        if (A.type == PETSc.Mat.Type.MPIAIJ):
            # A is an assembled distributed global matrix
            self._is_local = PETSc.IS().createGeneral(self.dofmap.indices)
            self.Ai = A.createSubMatrices(self._is_local)[0]
            self.vec_global = self.dofmap.create_vector()
        elif (A.type == PETSc.Mat.Type.SEQAIJ):
            # A is a sequential local matrix for each process
            self.vec_global = self.dofmap.create_vector()
            self.Ai = A
        else:
            raise Exception('Wrong matrix type')

        # Declare some variables, but leave the initialization
        # for setup step
        self.Ri = PETSc.Mat().create(MPI.COMM_SELF)
        self.Di = PETSc.Mat().create(MPI.COMM_SELF)
        self.solver = PETSc.KSP().create(MPI.COMM_SELF)
        self.scatter_type = ScatterType.PETSc

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
            self.solver.setOptionsPrefix("localsolver_")
            self.solver.setOperators(self.Ai)
            self.solver.setType('preonly')
            self.solver.pc.setType('lu')
            # self.solver.pc.setFactorSolverType('mumps')
            self.solver.setFromOptions()

        if self.scatter_type == ScatterType.PETSc:
            self.scatter = PETScVectorScatter(self.comm, self.dofmap)
        elif self.scatter_type == ScatterType.RMA:
            self.scatter = RMAVectorScatter(self.comm, self.dofmap)
        elif self.scatter_type == ScatterType.SHM:
            raise RuntimeError("Not implemented")

        # Define state of preconditioner, True means ready to use
        self.state = True

    @classmethod
    def global_matrix(obj, ASM) -> PETSc.Mat:
        '''
        Return the unassembled global matrix of type python.
        '''
        if ASM.state:
            # Define global unassembled global Matrix A
            A = PETSc.Mat().create()
            A.setSizes(ASM.sizes)
            A.setType(A.Type.PYTHON)
            A.setPythonContext(ASM)
            A.setUp()
            return A
        else:
            raise Exception('Matrix in wrong state.'
                            'Please setUP preconditioner first.')

    @property
    def type(self):
        return self.pc_type

    @property
    def sizes(self):
        N_owned = self.dofmap.size_owned
        Ng = self.dofmap.size_global
        return ((N_owned, Ng), (N_owned, Ng))

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec):
        """
        y = Sum_i^N R Ai^-1 R^T xi

        Parameters
        ==========

        pc: This argument is not called within the function but it
            belongs to the standard way of calling a preconditioner.

        x : Global distributed vector to which the preconditioner is to be applied.

        y : Global distributed vector  that stores the result
            of the preconditioning operation.

        """

        # Operate only on local forms of ghosted vectors
        with x.localForm() as x_local:
            if not x_local:
                raise RuntimeError('X vector is not ghosted')
            # Update ghosts and overlap dofs
            self.scatter.reverse(x_local, x)
            with y.localForm() as y_local:
                if not y_local:
                    raise RuntimeError('Y vector is not ghosted')

                if self.type == SMType.restricted:
                    work_vec = x_local.duplicate()
                    self.solver.solve(x_local, work_vec)
                    self.Di.mult(work_vec, y_local)
                elif self.type == SMType.additive:
                    self.solver.solve(x_local, y_local)
                else:
                    raise RuntimeError("Not implemented")

    def mult(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec):
        # y <- Ax

        # Operate only on local forms of vectors
        with x.localForm() as x_local:
            with y.localForm() as y_local:
                self.scatter.reverse(x_local, x)
                self.Ai.mult(x_local, y_local)

    @staticmethod
    def restritction_matrix(dofmap: DofMap) -> PETSc.Mat:
        """
        Explicitely construct the local restriction matrix for
        the current subdomain.
        Good for testing.
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
        Good for testing.
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
