import symmetry_invariant_operators as siop
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

symm_ops = siop.get_site_operations(species, coords) # Gives atom and s orb symms
print(symm_ops)
print("\n symmetry operations: \n", symm_ops)

#print("\nRandom 1-body Hamiltonian: \n", random_H1(symm_ops))
onebody_terms = siop.onebody_basis(symm_ops)
print("\n1-body basis: \n", onebody_terms)

#print("\nRandom 2-body Hamiltonian: \n", random_H2(symm_ops))
twobody_terms = siop.twobody_basis(symm_ops)
print("\n2-body basis size: \n", len(twobody_terms))
print("\n2-body basis: \n", twobody_terms)