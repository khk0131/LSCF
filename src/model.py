from scf_init import SCFInitGuess
from functional import Functional
from scipy.linalg import eigh
from pyscf import gto, dft
import jax.numpy as jnp

class SelfConsistentField(SCFInitGuess):
    """ Class for self consisten field calculation with jax
    """
    core_hamiltonian: jnp.array
    overlap_matrix: jnp.array
    orthogonalozation_matrix: jnp.array
    electron_repulsion_matrix: jnp.array
    mol: gto.Mole

    def __init__(self, 
                 mol_name: str='h2',
                 basis: str='sto-3g',
                 xc_f: str='LDA',
                 max_iter=15, 
                 energy_tolerance=1e-8, 
                 spin_restricted=True):
        self.mol_name = mol_name
        self.basis = basis
        self.xc_f = xc_f
        self.max_iter = max_iter
        self.energy_tolerance = energy_tolerance
        self.spin_restricted = spin_restricted
        self.core_hamiltonian = None
        self.overlap_matrix = None
        self.orthogonalozation_matrix = None
        self.electron_repulsion_matrix = None
        self.mol = None
        self.functional = Functional()

    def scf_loop(self):
        """ SCF iterations
            Density matrix will be ontained to calculate total energy further
        """
        # H_CORE, S = self.get_h_core_and_s(self.mol_name, self.basis)
        self.set_mol(self.mol_name, self.basis)
        P_0 = self.get_initial_density(n_electrons=self.mol.nelectron)
        F_0 = self.fock_matrix(P_0)
        
        for loop in range(self.max_iter):
            if loop == 0:
                F = F_0
            coefficient_matrix, _ = self.diagonalize_fock_matrix(F)
            C = self.orthogonalization_matrix @ coefficient_matrix
            P = self.denisty_matrix(C, n_electrons=self.mol.nelectron)
            F = self.fock_matrix(P)
        
        return P
        
    def fock_matrix(self, density_matrix):
        """ Construct fock matrix
        """
        assert self.core_hamiltonian is not None, "Please define core hamiltonian before contructing fock matrix and scf calculation"
        P = density_matrix
        if not self.spin_restricted:
            H_core = jnp.stack([self.core_hamiltonian for _ in range(2)], axis=0)
        else:
            H_core = self.core_hamiltonian
        J = self.get_electron_repulsion_matrix()
        J = jnp.einsum('ijkl, kl->ij', J, P)
        E_xc, V_xc = self.functional(mol=self.mol, density_matrix=P, xc_f=self.xc_f)
        fock_matrix = H_core  + J + V_xc
        return fock_matrix

    def diagonalize_fock_matrix(self, fock_matrix):
        """ Diagonalize fock matrix
        """
        F = self.orthogonalization_matrix.T @ fock_matrix @ self.orthogonalization_matrix
        eigvals, eigvecs = eigh(F)
        return eigvecs, eigvals

    def denisty_matrix(self, coefficient_matrix, n_electrons):
        """ Construct density matrix
        """
        C_occ = coefficient_matrix[:, :n_electrons]
        P = 2 * C_occ @ C_occ.T
        return P

    def get_total_energy(self, P):
        """ Return energy with KS eq
        """
        H_core = self.get_core_hamiltonian()
        J = self.get_electron_repulsion_matrix()
        E_nuc = self.get_nuclear_energy()
        E_xc, V_xc = self.functional(mol=self.mol, density_matrix=P, xc_f=self.xc_f)
        E = ((H_core + 0.5*J) * P).sum() + E_xc + E_nuc
        return E

# SCF for H2 under 6-31G(d) basis
model = SelfConsistentField(mol_name='h2', basis='6-31G(d)', max_iter=6)
P = model.scf_loop()
print("predicted energy =", model.get_total_energy(P))

# Test PySCF
mol = gto.Mole()
mol.atom = [['H',(1, 0, 0)], ['H',(0, 1, 0)]]
mol.basis = '6-31G(d)'
