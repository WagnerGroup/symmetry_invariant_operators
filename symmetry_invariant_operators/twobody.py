# Copyright (c) 2024 Sonali Joshi
# MIT License ( see "LICENSE")

import numpy as np

def symmetrize_twobody(O:np.ndarray, symm_ops:np.ndarray):
    """
    Performs symm_op.T @ O @ symm_op and averages for length of symm_ops. 

    Args: 
    O: a four-index (two-body) operator, shape [nsites, nsites, nsites, nsites]
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns: 
    O_symm : a symmetrized version of O, shape [nsites, nsites, nsites, nsites]
    """
    O_symm = np.zeros_like(O)
    for operation in symm_ops:
        O_symm += np.einsum('ai,bj,ck,dl, ijkl -> abcd', operation, operation, operation, operation, O)
    O_symm/=symm_ops.shape[0]
    return O_symm

def twobody_basis(symm_ops:np.ndarray):
    """
    Takes each element of two-body Hamiltonian and symmetrizes it to find
    symmetric invariant operators.
    Only the unique operators are returned.

    Args: 
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Basis of two-body operators invariant to the symm_ops. 
    Shape [N, nsites, nsites, nsites, nsites] where N changes based on symm_ops
    used.
    """
    N = symm_ops.shape[1]
    Asymm_list =[]
    for i in range(N):
        for j in range(i+1): # Only need to go over upper triangle elements
            for k in range(i+1):
                for l in range(i+1): 
                    A = np.zeros((N,N,N,N))
                    A[i,j,k,l]+= 0.25 # Intial Guess
                    A[l,k,j,i]+= 0.25 # Hermitian
                    A[k,l,i,j]+= 0.25 # Spin exchange symm (time-reversal symm for fermions)
                    A[j,i,l,k]+= 0.25 # Hermitian
                    Asymm = symmetrize_twobody(A, symm_ops)
                    found=np.allclose(Asymm, np.zeros_like(Asymm))
                    Asymm/=Asymm.max()
                    for As in Asymm_list:
                        if np.allclose(As, Asymm):
                            found=True
                            break
                    if not found:
                        if not np.allclose(Asymm, Asymm.T):
                            raise Exception("Did not produce a Hermitian Asymm")
                        Asymm_list.append(Asymm)
    return Asymm_list