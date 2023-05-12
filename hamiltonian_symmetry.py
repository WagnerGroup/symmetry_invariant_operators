import numpy as np

from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

def get_site_symm_ops(species, geom, center = True):
    """
    Args: 
    species: list of atom types ex: ["C", "H", "H"]
    geom: list of atom positions ex np.array([[0.,0.,0.],...])
    center: whether to center the molecule around the center of mass. This often will be the most symmetry.

    Returns: 
    site_symm_ops : a boolean ndarray of shape [nsymmops, natoms, natoms], with true where atom i is equivalent to atom j under 
    that particular symmetry operation
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

def symmetrize_onebody(O, symm_ops):
    """
    Args: 
    O: a two-index (one-body) operator
    symm_ops: symmetry operations

    Returns: 
    O_symm : a symmetrized version of O
    """
    O_symm = np.zeros_like(O)
    for operation in symm_ops:
        O_symm += np.einsum('ai,bj, ij -> ab', operation, operation, O)
    O_symm/=symm_ops.shape[0]
    return O_symm


def random_H1(symm_ops):
    """
    This function takes in a system's symmetry operators to constructs a random
    hermitian 1-body Hamiltonian.

    Args: 
    symm_ops: symmetry operations

    Returns:
    Random, symmetric one-body hamiltonian
    """
    N = symm_ops.shape[1]
    H = np.random.randn(N,N)
    H = 0.5*(H+H.T) # Hermitian
    return symmetrize_onebody(H, symm_ops)


def onebody_symm_basis(symm_ops):
    """
    Takes each element of one-body Hamiltonian and symmetrizes it to find
    symmetric invariant operators.
    Only the unique operators are returned.

    Args: 
    symm_ops: symmetry operations

    Returns:
    Basis of symmetric invariant one-body operators
    """
    N = symm_ops.shape[1]
    Asymm_list =[]
    for i in range(N):
        for j in range(N):
            A = np.zeros((N,N))
            A[i,j]=1.0
            Asymm = symmetrize_onebody(A, symm_ops)
            found=False
            for As in Asymm_list:
                if np.allclose(As, Asymm):
                    found=True
                    break
            if not found:
                Asymm_list.append(Asymm)
    return Asymm_list

def return_symm_inv_ops_H1( geom, symm_ops, return_rand_H = True ):
    '''
    This function takes in a geometry to constructs a random hermitian 1-body 
    Hamiltonian.
    The symmetry operators are then performed on the random Hamiltonian to
    find a Hamiltonian invariant to the symmetry operators given.

    Args: 
    geom : the cartesian coordinates of atoms, 2D array, Nx3
    symm_ops : float, 
    return_rand_H : boolean , changes outputs

    Returns:
    symm_inv_ops :
    rand_H : , only returned if conjugate input is True
    '''

    N = geom.shape[0]
    H = np.random.randn(N,N)
    rand_H = 0.5*(H+H.T) # Hermitian

    rand_symm_H = np.zeros((N,N))
    for i in np.arange(symm_ops.shape[0]):
        rand_symm_H += np.einsum('ai,bj, ij -> ab', symm_ops[i], symm_ops[i], rand_H)
    rand_symm_H /= symm_ops.shape[0]

    rounded_symm_H = np.round(rand_symm_H, decimals=4) # floating point error solution
    uni_vals = np.unique(rounded_symm_H)

    symm_inv_ops = []
    for n in np.arange(len(uni_vals)):
        i, j = np.where( rounded_symm_H == uni_vals[n] )
        symm_inv_ops.append( np.array((i,j)).T )
    symm_inv_ops = np.array(symm_inv_ops)

    if return_rand_H:
        return symm_inv_ops, rand_symm_H

    return symm_inv_ops

def return_symm_inv_ops_H2( geom, symm_ops, return_rand_H = True ):
    '''
    This function takes in a geometry to constructs a random hermitian 2-body 
    Hamiltonian.
    The symmetry operators are then performed on the random Hamiltonian to
    find a Hamiltonian invariant to the symmetry operators given.

    Args:
    geom : the cartesian coordinates of atoms, 2D array, Nx3
    symm_ops : float, 
    return_rand_H : boolean , changes outputs

    Returns:
    symm_inv_ops :
    rand_H : , only returned if conjugate input is True
    '''

    N = geom.shape[0]
    H = np.random.randn(N,N,N,N)
    rand_H = 0.5*(H+H.T) # Hermitian

    rand_symm_H = np.zeros((N,N,N,N))
    for i in np.arange(symm_ops.shape[0]):
        rand_symm_H += np.einsum('ai,bj,ck,dl,ijkl -> abcd', symm_ops[i], symm_ops[i], symm_ops[i], symm_ops[i], rand_H)
    rand_symm_H /= symm_ops.shape[0]

    rounded_symm_H = np.round(rand_symm_H, decimals=4) # floating point error solution
    uni_vals = np.unique(rounded_symm_H)

    symm_inv_ops = []
    for n in np.arange(len(uni_vals)):
        i, j, k, l = np.where( rounded_symm_H == uni_vals[n] )
        symm_inv_ops.append( np.array((i,j,k,l)).T )
    symm_inv_ops = np.array(symm_inv_ops)

    if return_rand_H:
        return symm_inv_ops, rand_symm_H

    return symm_inv_ops


if __name__ == "__main__":

    species = ["H", "H", "H", "H"]

    coords = [[0. , 0. , 0. ],
              [3.8, 0. , 0. ],
              [0. , 3.8, 0. ],
              [3.8, 3.8, 0. ]
             ]
    coords = np.array(coords)

    symm_ops = get_site_symm_ops(species, coords) # Gives atom and s orb symms

    print("symmetry operations", symm_ops)
    print("Random 1-body Hamiltonian: ", random_H1(symm_ops))
    print("1-body basis: ", onebody_symm_basis(symm_ops))