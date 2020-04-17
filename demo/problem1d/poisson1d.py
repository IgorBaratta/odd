# Copyright (C) 2020 Igor A. Baratta
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import numpy
import matplotlib.pyplot as plt
from cycler import cycler

import matplotlib.style
import matplotlib as mpl
mpl.style.use('seaborn')
mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrc')

U = numpy.load('sol.npy', allow_pickle=True)
X = numpy.load('points.npy', allow_pickle=True)

for j in range(0, 100, 5):
    for i in range(4):
        plt.plot(X[i], U[j][i])


plt.ylabel('some numbers')
plt.grid(True)
plt.tight_layout()
plt.show()

