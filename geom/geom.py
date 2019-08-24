import sys
import numpy as np
from scipy.linalg import eigh
import molecule

# read coordinate
input_data = np.loadtxt(sys.argv[-1], comments='#')
mol = molecule.Molecule(input_data)
print('***Cartesian Coordinates (a.u.)***')
mol.print_coord()

# build internal coordinates matrix
print('***Distance Matrix (a.u.)***')
mol.internal_coord = np.zeros( (mol.natom, mol.natom) )
for i in range(mol.natom):
  for j in range(i+1, mol.natom):
    mol.internal_coord[i, j] = mol.bond_length(i, j)

# print the list of bond angles(bond length < 4.0 a.u.), no need to save it
for i in range(mol.natom):
  for j in range(i+1, mol.natom):
    for k in range(j+1, mol.natom):
      if np.less([mol.bond_length(i,j),mol.bond_length(j,k)], [4.0,4.0]).all():
        print('A(',i,',', j,',' k,')', mol.bond_angle(i, j, k)*180.0/np.pi)

# print dihedral angles
for i in range(mol.natom):
  for j in range(i+1, mol.natom):
    for k in range(j+1, mol.natom):
      for l in range(k+1, mol.natom):
        if np.less([mol.bond_length(i,j),mol.bond_length(j,k),mol.bond_length(k,l)],
                     [4.0,4.0,4.0]).all():
          print('D(',i,',', j,',' k, ',', l,')', mol.dihedral_angle(i, j, k, l)*180.0/np.pi)

# print moment of inertia tensor
print(mol.mom_i_tensor)

mol.pri_mom_i, v = eigh(mol.mom_i_tensor)
mol.pri_mom_i.sort()
