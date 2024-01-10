from invariant_operators import ( get_site_symm_ops,
                                  onebody_symm_basis, 
                                  twobody_symm_basis )
import numpy as np

from rich import print

'''
Test with the H_4 molecule which has D4h symmetry.
'''

species = ["H", "H", "H", "H"]

coords = [[0. , 0. , 0. ],
          [3.8, 0. , 0. ],
          [0. , 3.8, 0. ],
          [3.8, 3.8, 0. ]]
coords = np.array(coords)

symm_ops = get_site_symm_ops(species, coords) # Gives atom and s orb symms
print(symm_ops)
print("\n symmetry operations: \n", symm_ops)

#print("\nRandom 1-body Hamiltonian: \n", random_H1(symm_ops))
print("\n1-body basis: \n", onebody_symm_basis(symm_ops))

#print("\nRandom 2-body Hamiltonian: \n", random_H2(symm_ops))
print("\n2-body basis size: \n", len(twobody_symm_basis(symm_ops)))
print("\n2-body basis: \n", twobody_symm_basis(symm_ops))