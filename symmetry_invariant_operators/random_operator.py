# Copyright (c) 2024 Sonali Joshi
# MIT License ( see "LICENSE")

import numpy as np

from onebody import symmetrize_onebody
from twobody import symmetrize_twobody

def random_H1(symm_ops:np.ndarray) -> np.ndarray:
    """
    This function takes in a system's symmetry operators to constructs a random
    hermitian 1-body Hamiltonian.

    Args: 
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Random, one-body Hamiltonian invariant to the symm_ops given. Shape [nsites, nsites]
    """
    N = symm_ops.shape[1]
    H = np.random.randn(N,N)
    H = 0.5*(H+H.T) # Hermitian
    return symmetrize_onebody(H, symm_ops)

def random_H2(symm_ops:np.ndarray) -> np.ndarray:
    """
    This function takes in a system's symmetry operators to constructs a random
    hermitian 2-body Hamiltonian.

    Args: 
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Random, two-body Hamiltonian invariant to the symm_ops given. Shape [nsites, nsites, nsites, nsites]  
    """
    N = symm_ops.shape[1]
    H = np.random.randn(N,N,N,N)
    H = 0.5*(H+H.T) # Hermitian
    return symmetrize_twobody(H, symm_ops)