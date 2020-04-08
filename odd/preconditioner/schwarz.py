# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of odd
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from enum import Enum

from odd import IndexMap


class SMType(Enum):
    """ TODO : class docstring """

    restricted = 1
    additive = 2
    multiplicative = 3


class AdditiveSchwarz(object):
    """ TODO : class docstring """

    def __init__(self, index_map: IndexMap):
        pass
