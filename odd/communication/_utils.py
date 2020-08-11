import numpy
import numba
import odd

from mpi4py import MPI


@numba.jit(nopython=True)
def global_to_local_numba(array, local_range, ghosts):
    output = numpy.empty_like(array)
    for i in range(array.size):
        if array[i] >= local_range[0] and array[i] < local_range[1]:
            output[i] = array[i] - local_range[0]
        else:
            new_index = numpy.where(ghosts == array[i])[0][0]
            output[i] = new_index + local_range[1] - local_range[0]
    return output
