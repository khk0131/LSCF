from scf_init import SCFInitGuess
from pyscf import gto
import jax.numpy as jnp

class SelfConsistentField(SCFInitGuess):
    """ Class for self consisten field calculation with jax
    """
    core_hamiltonian: jnp.array
    overlap_matrix: jnp.array
    electron_repulsion_matrix: jnp.array
    mol: gto.Mole

    def __init__(self, max_iter, energy_tolerance=1e-8, spin_restricted=False):
        self.max_iter = max_iter
        self.energy_tolerance = energy_tolerance
        self.spin_restricted = spin_restricted
        self.core_hamiltonian = None
        self.overlap_matrix = None
        self.electron_repulsion_matrix = None
        self.mol = None

    def __call__(self, initial_density_matrix):
        pass

    def scf_loop(self, initial_density_matrix):
        pass

    def fock_matrix(self, density_matrix):
        """ Construct fock matrix
        """
        assert self.core_hamiltonian is not None, "Please define core hamiltonian before contructing fock matrix and scf calculation"
        P = density_matrix
        if not self.spin_restricted:
            H_core = jnp.stack([self.core_hamiltonian for _ in range(2)], axis=0)
        J = self.get_electron_repulsion_matrix()
        J = jnp.einsum('ijkl, kl->ij', J, P)

        # F = H_core + J + V_xc
        fock_matrix = H_core  + J
        return fock_matrix

# SCF for H2 under 6-31G(d) basis
model = SelfConsistentField(max_iter=50)
P_0 = model.get_initial_density(*model.get_h_core_and_s(mol_name='h2', basis='6-31G(d)', num_basis=4), n_electrons=2)
print(model.mol)
print(model.fock_matrix(P_0))

# Test PySCF
mol = gto.Mole()
mol.atom = [['H',(1, 0, 0)], ['H',(0, 1, 0)]]
mol.basis = 'sto-3g'
mol.build()