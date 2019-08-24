import sys
import numpy as np
from scipy.linalg import eigh
import molecule

# read coordinate
input_geom = np.loadtxt(sys.argv[-1], comments='#')
mol = molecule.Molecule(input_geom)
print('***Cartesian Coordinates (a.u.)***')
mol.print_coord()

# read hessian
hess_file = sys.argv[-1].split("_")[0] + "_hessian.txt"
input_hess = np.loadtxt(hess_file, comments='#')
mol.hessian = input_hess.reshape( (mol.natom*3, mol.natom*3) )
print(mol.hessian)

# build mass weighted hessian
mol.mass_mat = np.repeat(mol.atms_mass, 3)
mol.mw_hess = np.einsum('ij,i,j->ij',
                        mol.hessian, mol.mass_mat**-.5, mol.mass_mat**-.5)

print(mol.mw_hess)

la, v = eigh(mol.mw_hess)
print(la)






