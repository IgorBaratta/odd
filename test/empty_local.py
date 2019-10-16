from petsc4py import PETSc
import numpy


class Matrix():
    def __init__(self):
        pass

    def mult(self, mat: PETSc.Mat, x: PETSc.Vec, y: PETSc.Vec):
        x.ghostUpdate()
        with x.localForm() as x_local:
            assert(x_local.handle)
            with y.localForm() as y_local:
                assert(y_local.handle)
                y_local.array[:] = x_local.array[:]


class Preconditioner():
    def __init__(self):
        pass

    def apply(self, pc: PETSc.PC, x: PETSc.Vec, y: PETSc.Vec):
        x.ghostUpdate()
        y.ghostUpdate()
        with x.localForm() as x_local:
            assert(x_local.handle)
            with y.localForm() as y_local:
                assert(y_local.handle)
                y_local.array[:] = x_local.array[:]


da = PETSc.DMDA().create([5, 5], stencil_width=1, comm=PETSc.COMM_WORLD)
x = da.createNaturalVec()
ghosts = [i % x.size for i in range(x.owner_range[1], x.owner_range[1] + 4)]
x.setMPIGhost(ghosts)
x.setRandom()
y = x.duplicate()

A = da.createMat()
Actx = Matrix()
A.setType(A.Type.PYTHON)
A.setPythonContext(Actx)
A.setUp()

PC = Preconditioner()

ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
ksp.setOperators(A)
ksp.setFromOptions()
ksp.pc.setType('python')
ksp.pc.setPythonContext(PC)
ksp.setFromOptions()
ksp.solve(x, y)

print(y.array)
#
# print(y.array)
# print(x.array_r)

# v = PETSc.Vec().create()
# v.setType(PETSc.Vec.Type.MPI)
# v.setSizes((5, None))
# ghosts = [i % v.size for i in range(v.owner_range[1], v.owner_range[1] + 3)]
# v.setMPIGhost(ghosts)
# v.setArray(numpy.array(range(*v.owner_range)))
# v.ghostUpdate()
#
# with v.localForm() as v_loc:
#     print(v_loc.array)
#
# A = PETSc.Mat().create()
# Actx = Matrix()
# A.setSizes(v.getSizes())
# A.setType(A.Type.PYTHON)
# A.setPythonContext(Actx)
# A.setUp()
# print(A.getSizes())
#
# # y = v.duplicate()
# # A.mult(v, y)

#
# # solver = PETSc.KSP().create(PETSc.COMM_WORLD)
# # solver.setOperators(A)
# # solver.setType('gmres')
# # solver.setUp()
# # solver.pc.setType('python')
# # solver.pc.setPythonContext(ASM)
# # solver.setFromOptions()
