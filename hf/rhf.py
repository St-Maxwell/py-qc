"""
input format is based on Psi4 input setting
variables in geometry are not supported
e.g.
molecule {
  O
  H 1 1.1
  H 1 1.1 2 104.
}
set basis cc-pVDZ
"""

import sys
import re
import numpy as np
from scipy import linalg
import psi4

with open(sys.argv[-1],'r') as input_file:
  input_text = input_file.read()

# read geometry and basis set from input file
geom_pattern = re.compile(r'molecule.*{((?:.|\n)*?)}')
basis_set_pattern = re.compile(r'set basis\s*(.*)')
input_geom = geom_pattern.findall(input_text)[0]
basis_set = basis_set_pattern.findall(input_text)[0]

mol = psi4.geometry(input_geom)
# psi4.set_options({'basis': 'def2-SVP',
#                      'scf_type': 'pk',
#                      'e_convergence': 1e-8})
wfn = psi4.core.Wavefunction.build(mol, basis_set)
mints = psi4.core.MintsHelper(wfn.basisset())
ndocc = wfn.nalpha()

# get integrals from psi4 module
Enuc = mol.nuclear_repulsion_energy()
S = np.asarray(mints.ao_overlap())
V = np.asarray(mints.ao_potential())
T = np.asarray(mints.ao_kinetic())
I = np.asarray(mints.ao_eri())

# diagonalize the overlap matrix
vec_s, U = linalg.eigh(S)
mat_s = np.diag( np.power(vec_s, -0.5) )
X = U.dot(mat_s).dot(U.T)
# vec_s, U = linalg.eigh(S)
# mat_s = np.diag( np.power(vec_s, -0.5) )
# X = U.dot(mat_s)

#
H = T + V
F = X.T.dot(H).dot(X)
vec_e, C = linalg.eigh(F)
C = X.dot(C)

C_occ = C[:, :ndocc]
D_old = np.einsum('pi,qi->pq', C_occ, C_occ)
Eele_old = np.einsum('ij,ij->', D_old, H+F)

# SCF
E_conv = 1.0E-6
D_conv = 1.0E-3
scf_max_cycle = 50
for scf_iter in range(scf_max_cycle):

    J = np.einsum('ij,pqji->pq', D_old, I)
    K = np.einsum('ij,pijq->pq', D_old, I)
    F = H + 2.0 * J - K
    
    F = X.T.dot(F).dot(X)
    vec_e, C = linalg.eigh(F)
    C = X.dot(C)
    C_occ = C[:, :ndocc]
    D_new = np.einsum('pi,qi->pq', C_occ, C_occ)
    Eele_new = np.einsum('ij,ij->', D_new, H+F)

    delta_E = np.abs(Eele_new - Eele_old)
    RMS_D = np.sqrt( np.sum( (D_new-D_old)**2 ) )
    if (delta_E < E_conv) and (RMS_D < D_conv):
        break
    
    Eele_old = Eele_new
    D_old = D_new

    if scf_iter == scf_max_cycle:
        psi4.core.clean()
        raise Exception("Maximum number of SCF cycles exceeded.")

Etot = Eele_new + Enuc
print('Final SCF energy: %.8f a.u.' % Etot)
# E_psi = psi4.energy('SCF')
# print('Psi4 SCF energy: %.8f a.u.' % E_psi)