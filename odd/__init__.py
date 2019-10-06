# -*- coding: utf-8 -*-
# Copyright (C) 2019 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Main module for odd"""

# Initialise PETSc/MPI
from dolfin import cpp
import sys
cpp.common.SubSystemsManager.init_logging(sys.argv)
del sys
cpp.common.SubSystemsManager.init_petsc()
del cpp

# Import public interface
from .DofMap import DofMap
from .Schwarz import AdditiveSchwarz, SMType
from .SubDomain import SubDomainData
from .VectorScatter import PETScVectorScatter, RMAVectorScatter
