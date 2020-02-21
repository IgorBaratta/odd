# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from enum import Enum
from mpi4py import MPI
from petsc4py import PETSc
from .subdomain import SubDomainData
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
        self.dt = data

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
            raise Exception('Matrix type not supported.')

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
            self.Ri = self.dt.restritction_matrix()
        if not self.Di.isAssembled():
            self.Di = self.dt.partition_of_unity("owned")

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
        y.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
