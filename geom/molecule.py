import numpy as np

h = 6.62607015E-34
c = 2.99792458E8
amu2kg = 1.66053878E-27 #mass
b2m = 0.529177249E-10 #length
au2J = 4.35974434E-18 #energy

atom_symbal = np.array([
  'H',                                    'He',
  'Li', 'Be', 'B',  'C',  'N', 'O', 'F',  'Ne',
  'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'
])
atom_mass = np.array([
  1.008,                                           4.003,
  6.941, 9.012, 10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
  22.99, 24.31, 26.98, 28.09, 30.97, 32.07, 35.45, 39.95
])

def is_unique(*numbers):
  if np.unique(numbers).size == len(numbers):
    return True
  return False

class Molecule(object):
  def __init__(self, input_data):
    self.natom = input_data[:,0].size
    self.atoms = np.int_(input_data[:,0])
    self.atms_mass = atom_mass[self.atoms - 1]
    self.atms_symb = atom_symbal[self.atoms - 1]
    self.mass = np.sum(self.atms_mass)
    self.coord = input_data[:,1:]
    self.cent_of_mass = (self.atms_mass[:,np.newaxis]*self.coord).sum(axis=0) / self.mass

  def vec_mode(self, vec):
    return np.sqrt( np.power(vec, 2).sum() )

  def bond_length(self, i, j):
    return self.vec_mode( self.coord[i,:] - self.coord[j,:] )

  def bond_angle(self, i, j, k):
    if not is_unique(i, j, k):
      raise ValueError('only three different atoms(i, j, k) are accepted.')
    vec1 = self.coord[i,:] - self.coord[j,:]
    vec2 = self.coord[k,:] - self.coord[j,:]
    return np.arccos( vec1.dot(vec2) / (self.vec_mode(vec1) * self.vec_mode(vec2)) )

  def dihedral_angle(self, i, j, k, l):
    if not is_unique(i, j, k, l):
      raise ValueError('only four different atoms(i, j, k, l) are accepted.')
    vec1 = self.coord[j,:] - self.coord[i,:]
    vec2 = self.coord[k,:] - self.coord[j,:]
    vec3 = self.coord[l,:] - self.coord[k,:]
    n1 = np.cross(vec1, vec2)
    n2 = np.cross(vec2, vec3)
    di_ang = np.arccos( n1.dot(n2) / (self.vec_mode(n1) * self.vec_mode(n2)) )
    vec4 = np.cross(n2, vec2)
    if ( n1.dot(vec4) > 0):
      return di_ang
    return -di_ang

  def translate(self, vec):
    self.coord += vec

  def print_coord(self):
    for i  in range(self.natom):
      print(' %s\t%.6f\t%.6f\t%.6f' % (self.atms_symb[i], self.coord[i,0], self.coord[i,1], self.coord[i,2]))