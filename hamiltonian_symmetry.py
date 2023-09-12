import numpy as np
from typing import List

from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

import h5py

def get_site_symm_ops(species:List[str], geom:np.ndarray, center:bool = True) -> np.ndarray:
    """
    Grabs point group symmetry of an arrangement of atoms. Uses pymatgen to grab
    the cartesian symmetry operators for said point group. Then converts the
    cartesian symmetry operators to the site based symmetry operators where each
    site is located at each atom.

    Args: 
    species: list of atom types ex: ["C", "H", "H"]
    geom: list of atom positions ex np.array([[0.,0.,0.],...])
    center: whether to center the molecule around the center of mass. This often
            will be the most symmetry.

    Returns: 
    site_symm_ops : a boolean ndarray of shape [nsymmops, natoms, natoms], with 
                    true where atom i is equivalent to atom j under that 
                    particular symmetry operation
    """
    
    mol = Molecule(species, geom)
    if center: 
        mol = mol.get_centered_molecule()
    mol_pga = PointGroupAnalyzer(mol)
    cart_symm_ops = mol_pga.get_symmetry_operations()
    site_symm_ops = np.zeros((len(cart_symm_ops), len(mol), len(mol)), dtype=bool)
    for k,operation in enumerate(cart_symm_ops): # in range(len(cart_symm_ops)):
        mol_symm = mol.copy()
        mol_symm.apply_operation(operation)
        for i, m in enumerate(mol):
            for j, n in enumerate(mol_symm):
                site_symm_ops[k,i,j] = m==n

        if not np.allclose(site_symm_ops[k]@site_symm_ops[k].T, np.identity(len(mol))):
            raise Exception("Did not produce a unitary transformation in get_site_symm")

    return site_symm_ops

def symmetrize_onebody(O:np.ndarray, symm_ops:np.ndarray):
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


def random_H1(symm_ops:np.ndarray):
    """
    This function takes in a system's symmetry operators to constructs a random
    hermitian 1-body Hamiltonian.

    Args: 
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Random, symmetric one-body hamiltonian with shape [nsites, nsites]
    """
    N = symm_ops.shape[1]
    H = np.random.randn(N,N)
    H = 0.5*(H+H.T) # Hermitian
    return symmetrize_onebody(H, symm_ops)


def onebody_symm_basis(symm_ops:np.ndarray):
    """
    Takes each element of one-body Hamiltonian and symmetrizes it to find
    symmetric invariant operators.
    Only the unique operators are returned.

    Args: 
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

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
                if np.allclose(As, Asymm):
                    found=True
                    break
            if not found:
                if not np.allclose(Asymm, Asymm.T):
                    raise Exception("Did not produce a Hermitian Asymm")
                Asymm_list.append(Asymm)
    return Asymm_list

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

def random_H2(symm_ops:np.ndarray):
    """
    This function takes in a system's symmetry operators to constructs a random
    hermitian 2-body Hamiltonian.

    Args: 
    symm_ops: symmetry operations with a shape [nsymmops, nsites, nsites]

    Returns:
    Random, two-body Hamiltonian invariant to the symm_ops given.  
    """
    N = symm_ops.shape[1]
    H = np.random.randn(N,N,N,N)
    H = 0.5*(H+H.T) # Hermitian
    return symmetrize_twobody(H, symm_ops)

def twobody_symm_basis(symm_ops:np.ndarray):
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

def save_symm_term_group(fname:str, symm_terms:np.ndarray):
    '''
    Args:
        fname : str
        symm_terms : 1-body or 2-body symm_terms given by onebody_symm_basis() 
        or twobody_symm_basis()
    
    Creates hdf5 file storing basis of symmetric terms in a minimal grouping. 
    Stores the number of sites, operator, indicies of sites.
    '''

    with h5py.File( fname, 'w') as f:
        f['minimal_groups'] = len(symm_terms)
        for i, symm_term in enumerate(symm_terms):
            index = np.array(np.where(symm_term != 0)).T
            f[f'group{i}/nsites']   = len(np.unique(index[0]))
            f[f'group{i}/operator'] = symm_term
            f[f'group{i}/indices']  = index
    return

def generate_group(current_list:np.ndarray, generators:np.ndarray):
    """
    current_list: list of symmetry operators currently in the set. Should be numpy 2D arrays. 
    generators: list of symmetry generators. Should be numpy 2D arrays. 

    EXAMPLE: c3v for a triangle
    generators = np.asarray( [ 
    [ #Identity
      [1,0,0],
      [0,1,0],
      [0,0,1],
    ],
    [ #rotation
      [0,1,0],
      [0,0,1],
      [1,0,0],
    ],
    [ #mirror
      [1,0,0],
      [0,0,1],
      [0,1,0],
    ],
    ])
    symmetry_operations = generate_group(generators, generators)
    """
    added_ops = []
    for op in current_list:
        for gen in generators:
            added_ops.append(gen@op)
    new_list = np.unique(np.vstack((current_list, np.asarray(added_ops))), axis=0)

    if new_list.shape[0] == current_list.shape[0]:
        return new_list
    return generate_group(new_list, generators)


if __name__ == "__main__":

    '''
    Test with the H_4 molecule which has D4h symmetry.
    '''

    species = ["H", "H", "H", "H"]

    coords = [[0. , 0. , 0. ],
              [3.8, 0. , 0. ],
              [0. , 3.8, 0. ],
              [3.8, 3.8, 0. ]
             ]
    coords = np.array(coords)

    symm_ops = get_site_symm_ops(species, coords) # Gives atom and s orb symms
    from rich import print
    print(symm_ops)
    print("\n symmetry operations: \n", symm_ops)

    print("\nRandom 1-body Hamiltonian: \n", random_H1(symm_ops))
    print("\n1-body basis: \n", onebody_symm_basis(symm_ops))

    print("\nRandom 2-body Hamiltonian: \n", random_H2(symm_ops))
    print("\n2-body basis size: \n", len(twobody_symm_basis(symm_ops)))