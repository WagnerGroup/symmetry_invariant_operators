import numpy as np

from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

def get_site_symm_ops(species, geom, center = True):
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

def symmetrize_onebody(O, symm_ops):
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


def random_H1(symm_ops):
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


def onebody_symm_basis(symm_ops):
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
            A[i,j]+= 0.5
            A[j,i]+= 0.5  # Normalized Hermitian
            Asymm = symmetrize_onebody(A, symm_ops)
            found=False
            for As in Asymm_list:
                if np.allclose(As, Asymm):
                    found=True
                    break
            if not found:
                if not np.allclose(Asymm, Asymm.T):
                    raise Exception("Did not produce a Hermitian Asymm")
                Asymm_list.append(Asymm)
    return Asymm_list

def symmetrize_twobody(O, symm_ops):
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

def random_H2(symm_ops):
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

def twobody_symm_basis(symm_ops):
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
                    A[i,j,k,l]+= 0.5
                    A[l,k,j,i]+= 0.5  # Normalized Hermitian
                    Asymm = symmetrize_twobody(A, symm_ops)
                    found=False
                    for As in Asymm_list:
                        if np.allclose(As, Asymm):
                            found=True
                            break
                    if not found:
                        if not np.allclose(Asymm, Asymm.T):
                            raise Exception("Did not produce a Hermitian Asymm")
                        Asymm_list.append(Asymm)
    return Asymm_list

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

    print("\n symmetry operations: \n", symm_ops)

    print("\nRandom 1-body Hamiltonian: \n", random_H1(symm_ops))
    print("\n1-body basis: \n", onebody_symm_basis(symm_ops))

    print("\nRandom 2-body Hamiltonian: \n", random_H2(symm_ops))
    print("\n2-body basis size: \n", len(twobody_symm_basis(symm_ops)))