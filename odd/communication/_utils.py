import numpy
import numba


@numba.jit(nopython=True)
def global_to_local_numba(array, local_range, ghosts):
    out = numpy.empty_like(array)
    for i in range(array.size):
        if array[i] >= local_range[0] and array[i] < local_range[1]:
            out[i] = array[i] - local_range[0]
        else:
            new_index = numpy.where(ghosts == array[i])[0][0]
            out[i] = new_index + local_range[1] - local_range[0]
    return out
