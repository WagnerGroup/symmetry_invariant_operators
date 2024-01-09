# Invariant Operators

invariant_operators is a package to generate a set of symmetry invariant operators given a finite group.

This can be used to constuct 2-body symmetry invariant Hamiltonians for a given system.

## 2-body Hamiltonian

Given a second quantized version of the 2-body Hamiltonian:
$$ \hat{H} = \sum_{i,j} t_{ij} \hat{c}_i^\dag \hat{c}_j^{} + \sum_{i,j,k,l} V_{ijkl} \hat{c}_i^{\dag} \hat{c}_j \hat{c}_k^\dag \hat{c}_l^{} $$
where $i,j,k,l$ are the indicies of the atoms.

And some set of symmetry operators $\{\hat{S}\}$ for the above Hamiltonian it is true that $ \hat{S} \hat{H} \hat{S} = \hat{H}$, it is invariant to the finite group of symmetry operators.

But with this tool we can further breakdown the operators found in the Hamiltonian in the a small set of operators that are invariant to $\{\hat{S}\}$. 

For example lets look at a $\text{H}_2$ molecule (2 atom system?):

The 1-body part of the Hamiltonian can breakdown into the following operators:
$$ \hat{O}_1 =  t_1 \sum_i \hat{c}_i^\dag \hat{c}_i^{} \text{ and }  \hat{O}_2 = t_2 \sum_{<i,j>} \hat{c}_i^\dag \hat{c}_j^{} $$
where both same atom $\hat{O}_1$ and next-nearest atom $\hat{O}_2$ are symmetry invariant to the given finite group.

This can also be done for the 2-body part of the Hamiltonian.

This package will then provide those symmetry invariant $\{\hat{O}\}$.
