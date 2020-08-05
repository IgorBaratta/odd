import odd
import numpy
import time

b = odd.ones(10000000)

t = time.time()
c = numpy.sin(b)
t = time.time() - t

if b._map.forward_comm.rank == 0:
    print(t)
# d = b + c

# nc = numpy.linalg.norm(c)
