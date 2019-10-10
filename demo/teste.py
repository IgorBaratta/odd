#!/usr/bin/env python
"""
Parallel PI computation using Remote Memory Access (RMA)
within Python objects exposing memory buffers (requires NumPy).
usage::
  $ mpiexec -n <nprocs> python cpi-rma.py
"""

from mpi4py import MPI
from math import pi as PI
from numpy import array
import sys


nprocs = MPI.COMM_WORLD.Get_size()
myrank = MPI.COMM_WORLD.Get_rank()

n = array([1, 2, 3, 4], dtype=int)
dist_unit = 8
win = MPI.Win.Create(n,  disp_unit=dist_unit, comm=MPI.COMM_WORLD)
buffer = array(-1, dtype=int)

win.Fence()
win.Get(buffer, 0, target=myrank)
win.Fence()

win.Free()

print(myrank, buffer)


# import numpy as np
# from mpi4py import MPI
#
#
# comm = MPI.COMM_WORLD
# size = comm.Get_size()
# rank = comm.Get_rank()
# status = MPI.Status()
#
#
# dim = 10
# itemsize = MPI.DOUBLE.Get_size()
# nbytes = dim*itemsize
#
# if rank == 0:
#     # memory = MPI.Alloc_mem(nbytes + 100)
#     memory = np.arange(dim).astype(np.float64)
#     print(memory)
#
#     fwin = MPI.Win.Create(memory, 1, comm=comm)
#
# else:
#     cfwin = MPI.Win.Create(None, comm=comm)
#     rbuf = np.zeros(dim, dtype=np.float64)
#     print(rank, rbuf)
#     print(rank, 'start get...')
#     cfwin.Lock(0, lock_type=235)    # MPI_LOCK_SHARED = 235
#     cfwin.Get(rbuf, target_rank=0)
#     # req = cfwin.Rget(rbuf, target_rank=0)
#     cfwin.Unlock(0)
#
#     # req.Wait()
#
#     print(rank, 'get complete.')
#     print(rank, rbuf)
