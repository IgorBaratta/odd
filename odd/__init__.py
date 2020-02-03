# Copyright (C) 2019 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Main module for odd"""

# Initialise PETSc/MPI
from dolfinx import cpp
import sys
cpp.common.SubSystemsManager.init_logging(sys.argv)
del sys
cpp.common.SubSystemsManager.init_petsc()
del cpp

# Import public interface
from .dofmap import DofMap
from .schwarz import AdditiveSchwarz, SMType
from .subdomain import SubDomainData
from .vector_scatter import PETScVectorScatter, RMAVectorScatter, ScatterType
from .assemble import assemble_vector
