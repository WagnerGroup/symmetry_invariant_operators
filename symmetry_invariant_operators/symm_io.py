# Copyright (c) 2024 Sonali Joshi
# MIT License ( see "LICENSE")

import numpy as np

import h5py

def save_to_hdf5(fname:str, symm_terms:np.ndarray):
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