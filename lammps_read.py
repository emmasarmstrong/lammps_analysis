#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:58:07 2022

@author: emmaarmstrong
"""

import numpy as np

#function to convert lammps bound box parameters to lattice parameters 
def lammpstolatt(bound):
    latt = np.zeros([3,3])
    latt[0,0] = bound[0,1] - bound[0,0] - abs(bound[0,2]) - abs(bound[1,2])
    latt[1,1] = bound[1,1] - bound[1,0] - abs(bound[2,2])
    latt[2,2] = bound[2,1] - bound[2,0]
    latt[1,0] = bound[0,2]
    latt[2,0] = bound[1,2]
    latt[2,1] = bound[2,2]
    return latt.T

#read lammps trajectory file - small enough file that readlines() is acceptable to use
def traj_read(filepath,filename,frames):
    f = open(filepath+filename,'r').readlines()
    #get number of atoms and bound box size from file
    n_atoms = int(f[3])
    bound = np.array([[float(x) for x in f[5].split()],[float(x) for x in f[6].split()],[float(x) for x in f[7].split()]])
    latt = lammpstolatt(bound)
    #atom data - atom index, atom type, xs, ys, zs
    coords = np.zeros([frames,n_atoms,5])
    for i in range(frames):
        for j in range(n_atoms):
            data = f[9+i*(9+n_atoms)+j].split()
            coords[i][j] = data[0:5]       
    return coords,n_atoms,latt

#identify molecules from bonded atoms in lammps data file
def mol_read(filepath,n_mols,bond_type):
    f = open(filepath+'data.lmp','r').readlines()
    mols = np.zeros([n_mols,3])
    #for three atom molecules can use angle data - will need impropers for four atom molecules or combine angle data
    angles = int(f[6].split()[0])
    for i in range(len(f)):
        if f[i] == 'Angles\n':
            counter = 0
            for j in range(angles):
                data = f[i+2+j].split()
                if data[1] == str(bond_type):
                   mols[counter] = data[2:]
                   counter += 1             
    return mols

x = np.array([0.0,7.9635801315,-7.9635797775])
y = np.array([0.0,5.7484013127,0.0])
z = np.array([0.0,4.9652400017,0.0])

l = lammpstolatt(np.array([x,y,z]))