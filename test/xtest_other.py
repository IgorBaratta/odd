#
# # Create Function Space and Dofmap
# V = FunctionSpace(mesh, ("Lagrange", 1))
#
# u, v = TrialFunction(V), TestFunction(V)
#
# a = inner(grad(u), grad(v)) * dx
# L = inner(1.0, v) * dx
# A = assemble_matrix(a)
# A.assemble()
#
# subdomains = Decompose(mesh, V, comm)
# subdomains.setUP("RAS")
#
#
# indices = V.dofmap.index_map.indices(True).astype(IntType)
# is_A = PETSc.IS().createGeneral(indices)
# A_local = A.createSubMatrix(is_A, is_A)
# # print(r.indices)
# # print(V.dofmap.index_map.indices(True))
# if comm.rank == 0:
#     print(A_local.getLocalSize())
#     print(A.getLocalSize())
#
#
# # Initialize local ksp solver.
# local_ksp = PETSc.KSP().create()
# local_ksp.setOperators(A_local)
# local_ksp.setType("preonly")
# local_PC = local_ksp.getPC()
# local_PC.setType("lu")
#
#
#
# # A_mpiaij = A.copy()
# # A_mpiaij_local = A_mpiaij.createSubMatrices(is_A)[0]
# # A_scaled = A.copy().getISLocalMat()
# # vglobal, _ = A.getVecs()
# # vlocal, _ = A_scaled.getVecs()
# # scatter_l2g = PETSc.Scatter().create(vlocal, None, vglobal, is_A)
#
#
#
# ff = MeshFunction("size_t", mesh, mesh.topology.dim - 1, 0)
# if subdomain.id == 0:
#     ff.values[:] = subdomain.id + 1
#
# enconding = XDMFFile.Encoding.HDF5
# with XDMFFile(mesh.mpi_comm(), "interface.xdmf", encoding=enconding) as xdmf:
#     xdmf.write(ff)



a1 = cpp.mesh.midpoints(mesh, 1, range(mesh.num_entities(1)))
a2 = cpp.mesh.midpoints(subdomain.mesh, 1, range(subdomain.mesh.num_entities(1)))

print(a1==a2)
