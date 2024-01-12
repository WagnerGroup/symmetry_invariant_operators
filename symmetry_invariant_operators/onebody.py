# Copyright (c) 2024 Sonali Joshi
# MIT License ( see "LICENSE")

import numpy as np
from typing import List

def symmetrize_onebody(O:np.ndarray, symm_ops:np.ndarray) -> np.ndarray:
    """
    Performs symm_op.T @ O @ symm_op and averages for length of symm_ops. 

    Args: 
    O: a two-index (one-body) operator, shape [nsites, nsites]
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns: 
    O_symm : a symmetrized version of O, shape [nsites, nsites]
    """
    O_symm = np.zeros_like(O)
    for operation in symm_ops:
        O_symm += np.einsum('ai,bj, ij -> ab', operation, operation, O)
    O_symm/=symm_ops.shape[0]
    return O_symm

def onebody_basis(symm_ops:np.ndarray, rtol: float=1e-5) -> List[np.ndarray]:
    """
    Takes each element of one-body Hamiltonian and symmetrizes it to find
    symmetric invariant operators.
    Only the unique operators are returned.

    Args: 
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]
    rtol: The relative tolerance parameter (see np.allclose notes)

    Returns:
    Basis of one-body operators invariant to the symm_ops. 
    Shape [N, nsites, nsites] where N changes based on symm_ops used.
    """
    N = symm_ops.shape[1]
    Asymm_list =[]
    for i in range(N):
        for j in range(i+1): # Only need to go over upper triangle elements
            A = np.zeros((N,N))
            A[i,j]+= 1.0
            A[j,i]+= 1.0  # Normalized Hermitian
            Asymm = symmetrize_onebody(A, symm_ops)
            found=np.allclose(Asymm, np.zeros_like(Asymm))
            Asymm/=Asymm.max()
            for As in Asymm_list:
                if np.allclose(As, Asymm, rtol=rtol) or np.allclose(As, -Asymm, rtol=rtol):
                    found=True
                    break
            if not found:
                if not np.allclose(Asymm, Asymm.T):
                    raise Exception("Did not produce a Hermitian Asymm")
                Asymm_list.append(Asymm)
    return Asymm_list