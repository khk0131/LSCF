import sys
sys.path.append('../src/')
from model import SelfConsistentField
from pyscf import gto

# SCF for H2 under 6-31G(d) basis
model = SelfConsistentField(mol_name='h2', basis='6-31G(d)', max_iter=6)
P = model.scf_loop()
print("predicted energy =", model.get_total_energy(P))

# Test PySCF
mol = gto.Mole()
mol.atom = [['H',(1, 0, 0)], ['H',(0, 1, 0)]]
mol.basis = '6-31G(d)'
# mf = dft.RKS(mol)        
# mf.xc = 'lda, vwn'
# mf.grids.level = 6
# mf.kernel()