from scipy import sparse as sp
from .index_map import IndexMap


class LinearOperator(object):
    """
    Base Class for Linear Operator
    """
    def __init__(self, local_mat: sp.spmatrix, index_map: IndexMap):
        """
        Ai is the local matrix, possibly with transmission condition
        Di is the partition of unity matrix
        """
        self.local_mat = local_mat
        self.index_map = index_map

        # Check if sizes match: