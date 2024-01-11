import numpy as np
from typing import List

from pymatgen.core import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

def get_site_operations(species:List[str], geom:np.ndarray, center:bool = True) -> np.ndarray:
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

def generate_group(current_list:np.ndarray, generators:np.ndarray, round:int=9, iteration:int=0, maximum_group_size:int=1000):
    """
        The group size for a given point group are well defined and should be 
        referenced via a table. The generated group size should be
        less than or equal to the point group size. 
        If the generated group size is less than the point group size, it should
        be checked that it is due to basis of the symmetry operators. 
        Example: s obitals symmetry operator for H4 molecule. 
        H4 molecule has D4H symmetry, but the generated group will have D4 
        symmetry as the s orbitals do not break sigma_h symmetry. 

        Args:
        current_list: list of symmetry operators currently in the set. Should be numpy 2D arrays.
        generators: list of symmetry generators. Should be numpy 2D arrays.
        round: number of digits to round to when checking for duplicates
        interation: iteration number, CAN be used if keeping track of interation
                    or killing after certain number of iterations
        maximum_group_size: will kill recoursion after the number of symmetry 
                            operators in current_list exceeds this value.
    ​
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
        symmetry_operations.shape[0] # group size
    """
    if current_list.shape[0] > maximum_group_size:
        raise ValueError("Group size is too large.")
    # generate new operators
    added_ops = []
    for op in current_list:
        for gen in generators:
            added_ops.append(gen @ op)
    added_ops = np.asarray(added_ops)

    # remove duplicates
    stack = np.vstack((current_list, np.asarray(added_ops)))
    new_list, new_inds = np.unique(
        np.round(stack, round), 
        axis=0, return_index=True
    )
    new_list = stack[new_inds]

    if new_list.shape[0] == current_list.shape[0]:
        return new_list
    return generate_group(new_list, generators, iteration=iteration+1)

def generate_rotation_p_orbitals(a:float, b:float, norb:int=3, tol:float=1e-4, basis_change=None):
    # http://openmopac.net/manual/rotate_atomic_orbitals.html
    # Basis ordering: {p_{x}, p_{y}, p_{z}}
    # Spherical coordinates: a = rotation angle about z in radians, b = rotation angle about x in radians
    if basis_change is None:
        basis_change = np.eye(norb)

    R_pi_plus_x_x  = np.cos(a) * np.cos(b)
    R_pi_minus_x_x = np.sin(a) * np.cos(b)
    R_sigma_x_x    = -np.sin(b)

    R_pi_plus_y_y  = -np.sin(a)
    R_pi_minus_y_y = np.cos(a)
    R_sigma_y_y    = 0

    R_pi_plus_z_z  = np.cos(a) * np.sin(b)
    R_pi_minus_z_z = np.sin(a) * np.sin(b)
    R_sigma_z_z    = np.cos(b)

    symmop = np.array(
        [
            [R_pi_plus_x_x , R_pi_plus_y_y , R_pi_plus_z_z ],
            [R_pi_minus_x_x, R_pi_minus_y_y, R_pi_minus_z_z],
            [R_sigma_x_x   , R_sigma_y_y   , R_sigma_z_z   ],
        ]
    )
    symmop = np.einsum("ij,ai,bj->ab", symmop, basis_change, basis_change)
    assert np.linalg.norm(np.einsum("ij,jk->ik", symmop.T, symmop) - np.eye(norb)) < tol # norm check

    return symmop

def generate_rotation_d_orbitals(a:float, b:float, norb:int=3, tol:float=1e-4, basis_change=None):
    # http://openmopac.net/manual/rotate_atomic_orbitals.html
    # Basis ordering: {d_{x^2-y^2}, d_{xz}, d_{z^2}, d_{yz}, d_{xy}}
    # Spherical coordinates: a = rotation angle about z, b = rotation angle about x
    if basis_change is None:
        basis_change = np.eye(norb)

    R_x2_y2_x2_y2 = (2 * (np.cos(a) ** 2) - 1) * (np.cos(b) ** 2) + 0.5 * (
        2 * (np.cos(a) ** 2) - 1
    ) * (np.sin(b) ** 2)
    R_x2_y2_xz = -np.cos(a) * np.sin(b) * np.cos(b)
    R_x2_y2_z2 = np.sqrt(3 / 4) * (np.sin(b) ** 2)
    R_x2_y2_yz = -np.sin(a) * np.sin(b) * np.cos(b)
    R_x2_y2_xy = 2 * np.sin(a) * np.cos(a) * (np.cos(b) ** 2) + np.sin(a) * np.cos(
        a
    ) * (np.sin(b) ** 2)

    R_xz_x2_y2 = (2 * (np.cos(a) ** 2) - 1) * np.sin(b) * np.cos(b)
    R_xz_xz = np.cos(a) * (2 * (np.cos(b) ** 2) - 1)
    R_xz_z2 = -np.sqrt(3) * np.sin(b) * np.cos(b)
    R_xz_yz = np.sin(a) * (2 * (np.cos(b) ** 2) - 1)
    R_xz_xy = 2 * np.sin(a) * np.cos(a) * np.sin(b) * np.cos(b)

    R_z2_x2_y2 = np.sqrt(3 / 4) * (2 * (np.cos(a) ** 2) - 1) * (np.sin(b) ** 2)
    R_z2_xz = np.sqrt(3) * np.cos(a) * np.sin(b) * np.cos(b)
    R_z2_z2 = (np.cos(b) ** 2) - 0.5 * (np.sin(b) ** 2)
    R_z2_yz = np.sqrt(3) * np.sin(a) * np.sin(b) * np.cos(b)
    R_z2_xy = np.sqrt(3) * np.sin(a) * np.cos(a) * (np.sin(b) ** 2)

    R_yz_x2_y2 = -2 * np.sin(a) * np.cos(a) * np.sin(b)
    R_yz_xz = -np.sin(a) * np.cos(b)
    R_yz_z2 = 0
    R_yz_yz = np.cos(a) * np.cos(b)
    R_yz_xy = (2 * (np.cos(a) ** 2) - 1) * np.sin(b)

    R_xy_x2_y2 = -2 * np.sin(a) * np.cos(a) * np.cos(b)
    R_xy_xz = np.sin(a) * np.sin(b)
    R_xy_z2 = 0
    R_xy_yz = -np.cos(a) * np.sin(b)
    R_xy_xy = (2 * (np.cos(a) ** 2) - 1) * np.cos(b)

    symmop = np.array(
        [
            [R_x2_y2_x2_y2, R_x2_y2_xz, R_x2_y2_z2, R_x2_y2_yz, R_x2_y2_xy],
            [R_xz_x2_y2, R_xz_xz, R_xz_z2, R_xz_yz, R_xz_xy],
            [R_z2_x2_y2, R_z2_xz, R_z2_z2, R_z2_yz, R_z2_xy],
            [R_yz_x2_y2, R_yz_xz, R_yz_z2, R_yz_yz, R_yz_xy],
            [R_xy_x2_y2, R_xy_xz, R_xy_z2, R_xy_yz, R_xy_xy],
        ]
    )
    symmop = np.einsum("ij,ai,bj->ab", symmop, basis_change, basis_change)
    assert np.linalg.norm(np.einsum("ij,jk->ik", symmop.T, symmop) - np.eye(norb)) < tol

    return symmop