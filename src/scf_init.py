from scipy.linalg import eigh
import numpy as np
import jax.numpy as jnp
from pyscf import gto
from constant import ANGSTROM_TO_BOHR

class SCFInitGuess:
    core_hamiltonian: jnp.array
    overlap_matrix: jnp.array
    electron_repulsion_matrix: jnp.array
    mol: gto.Mole

    def __init__(self):
        pass
    
    def get_initial_density(self, H_core, S, n_electrons):
        """ Get initial density matrix from core hamiltonian and overlap matrix
        """
        eigvals, eigvecs = eigh(S)
        S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        H_ortho = S_inv_sqrt.T @ H_core @ S_inv_sqrt
        
        eps, C_ortho = eigh(H_ortho)
        C = S_inv_sqrt @ C_ortho
        n_occ = n_electrons // 2
        C_occ = C[:, :n_occ]
        P = 2 * C_occ @ C_occ.T

        return P

    def get_h_core_and_s(self, mol_name='h2', basis='sto-3g', num_basis=4):
        """ Get initial core hamiltonian and overlap matrix 
            TODO: H_core and S should be calculated according to mol and basis
        """
        self.set_mol(mol_name, basis)
        H_core = np.random.rand(num_basis, num_basis)
        S = np.eye(num_basis)
        self.core_hamiltonian = H_core
        self.overlap_matrix = S

        return self.core_hamiltonian, self.overlap_matrix

    def get_electron_repulsion_matrix(self):
        """ Get electron repulsion matrix
        """
        J = self.mol.intor('int2e') # coulomb replusion
        return J

    def set_mol(self, mol_name, basis):
        """ Set molecular information with gto.M method
        """
        if mol_name == 'h2':
            atom_symbol = jnp.array([1, 1])
            nuc_pos = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

        self.mol = gto.M(
            atom=list(zip(atom_symbol, nuc_pos*ANGSTROM_TO_BOHR)),
            basis=basis,
            unit='Bohr'
        )
        


        
