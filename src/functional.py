import jax.numpy as jnp
from pyscf import gto, dft

class Functional:
    
    def __init__(self):
        pass

    def __call__(self, mol, density_matrix, xc_f='LDA'):
        if xc_f == 'LDA':
            return self.LDA(mol, density_matrix=density_matrix)
            
    def LDA(self, mol, density_matrix):
        """ E_xc and V_xc with LDA functional
        """
        mf = dft.RKS(mol)
        mf.xc = 'lda, vwn'
        mf.grids.level = 6

        ni = mf._numint
        nelec, excsum, vmat = ni.get_vxc(mol, mf.grids, mf.xc, dms=density_matrix)
        return excsum, vmat
        
# Test PySCF
mol = gto.Mole()
mol.atom = [['H',(1, 0, 0)], ['H',(0, 1, 0)]]
mol.basis = '6-31G(d)'
mol.build()
mol.kernel()

mf = dft.RKS(mol)        
mf.xc = 'lda, vwn'
mf.grids.level = 6
mf.kernel()