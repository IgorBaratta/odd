# Copyright (C) 2019 Garth N. Wells
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for custom Python assemblers"""

import ctypes
import ctypes.util
import os

import cffi
import numba
import numba.cffi_support
import numpy as np
from petsc4py import PETSc, get_config as PETSc_get_config

import dolfinx


petsc_dir = PETSc_get_config()['PETSC_DIR']

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

scalar_size = np.dtype(PETSc.ScalarType).itemsize
index_size = np.dtype(PETSc.IntType).itemsize

if index_size == 8:
    c_int_t = "int64_t"
    ctypes_index = ctypes.c_int64
elif index_size == 4:
    c_int_t = "int32_t"
    ctypes_index = ctypes.c_int32
else:
    raise RecursionError("Unknown PETSc index type.")

if complex and scalar_size == 16:
    c_scalar_t = "double _Complex"
    numba_scalar_t = numba.types.complex128
elif complex and scalar_size == 8:
    c_scalar_t = "float _Complex"
    numba_scalar_t = numba.types.complex64
elif not complex and scalar_size == 8:
    c_scalar_t = "double"
    numba_scalar_t = numba.types.float64
elif not complex and scalar_size == 4:
    c_scalar_t = "float"
    numba_scalar_t = numba.types.float32
else:
    raise RuntimeError(
        ("Cannot translate PETSc scalar type" +
         "to a C type, complex: {} size: {}.").format(complex, scalar_size))


# CFFI - register complex types
ffi = cffi.FFI()
numba.cffi_support.register_type(ffi.typeof('double _Complex'),
                                 numba.types.complex128)
numba.cffi_support.register_type(ffi.typeof('float _Complex'),
                                 numba.types.complex64)


# Get MatSetValuesLocal and MatSetValues from PETSc available via cffi in ABI mode
ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
                                  {0} ncol, const {0}* icol, const {1}* y, int addv);
""".format(c_int_t, c_scalar_t))

ffi.cdef("""int MatSetValues(void* mat, {0} m, const {0}* idxm,
                                  {0} n, const {0}* idxn, const {1}* v, int addv);
""".format(c_int_t, c_scalar_t))

petsc_lib_name = ctypes.util.find_library("petsc")
if petsc_lib_name is not None:
    petsc_lib_cffi = ffi.dlopen(petsc_lib_name)
else:
    try:
        petsc_lib_cffi = ffi.dlopen(os.path.join(petsc_dir, "lib", "libpetsc.so"))
    except OSError:
        petsc_lib_cffi = ffi.dlopen(os.path.join(petsc_dir, "lib", "libpetsc.dylib"))
    except OSError:
        print("Could not load PETSc library for CFFI (ABI mode).")
        raise

MatSetValuesLocal = petsc_lib_cffi.MatSetValuesLocal
MatSetValues = petsc_lib_cffi.MatSetValues

dolfinx.MPI.comm_world.barrier()
