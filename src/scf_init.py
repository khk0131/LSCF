from scipy.linalg import eigh
import numpy as np
import jax.numpy as jnp
from pyscf import gto, dft
from constant import ANGSTROM_TO_BOHR

class SCFInitGuess:
    core_hamiltonian: jnp.array
    overlap_matrix: jnp.array
    orthogonalization_matrix: jnp.array
    electron_repulsion_matrix: jnp.array
    nuclear_energy: jnp.float64
    mol: gto.Mole

    def __init__(self):
        pass
    
    def get_initial_density(self, n_electrons):
        """ Get initial density matrix from core hamiltonian and overlap matrix
        """
        eigvals, eigvecs = eigh(self.overlap_matrix) 
        self.orthogonalization_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
        S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        H_ortho = S_inv_sqrt.T @ self.core_hamiltonian @ S_inv_sqrt
        
        eps, C_ortho = eigh(H_ortho)
        C = S_inv_sqrt @ C_ortho
        n_occ = n_electrons // 2
        C_occ = C[:, :n_occ]
        P = 2 * C_occ @ C_occ.T

        return P

    def get_core_hamiltonian(self):
        """ Get core hamiltonian matrix 
        """
        return self.core_hamiltonian

    def get_overlap_matrix(self):
        """ Get overlap matrix
        """
        return self.overlap_matrix

    def get_electron_repulsion_matrix(self):
        """ Get electron repulsion matrix
        """
        return self.electron_repulsion_matrix

    def get_nuclear_energy(self):
        """ Get nuclear energy
        """
        return self.nuclear_energy

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
        mf = dft.RKS(self.mol)
        self.core_hamiltonian = mf.get_hcore()
        self.overlap_matrix = self.mol.intor('int1e_ovlp')
        self.electron_repulsion_matrix = self.mol.intor('int2e')
        self.nuclear_energy = self.mol.energy_nuc()
        


        
