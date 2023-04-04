#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:06:43 2022

@author: emmaarmstrong
"""

import numpy as np
from cell_distances_rdf import neighbours

#check distance between donor and acceptor atoms can be classified as a h bond
def dist_check(x0,x1,latt):
    r01 = x0 - x1
    r01 = r01 - np.rint(r01)
    if np.sqrt(np.sum(np.dot(latt,r01)**2)) <= 3.5:
        return 1
    else:
        return 0

#check angle between donor, hydrogen and acceptor atoms can be classified as a h bond    
def ang_check(x1,x0,x2,latt):
    s01 = x0 - x1
    s02 = x0 - x2
    s01 = s01 - np.rint(s01)
    s02 = s02 - np.rint(s02)
    r01 = np.dot(latt,s01)
    r02 = np.dot(latt,s02)
    v01 = r01 / np.linalg.norm(r01)
    v02 = r02 / np.linalg.norm(r02)
    #if np.arccos(np.dot(v01,v02)) >= np.radians(150):
    if np.arccos(np.dot(v01,v02)) <= np.radians(30):
        
        return 1
    else:
        return 0

#count h bonds in system, returns donor and acceptor indices
def h_bonding(n_cells,cells,coords,mols,latt):
    h_bonds = []
    #identify filled cells
    ix,iy,iz = np.nonzero(cells)
    n_filled = len(ix)
    for n in range(n_filled):
        c0 = cells[ix[n],iy[n],iz[n]]
        #identify neighbouring cells
        for ind in neighbours(ix[n],iy[n],iz[n],n_cells):
            c1 = cells[ind[0],ind[1],ind[2]]
            #check neighbouring cell is not empty
            if c1 != 0:
                #loop through donor and acceptor atoms in the cells
                for p in c0:
                    for q in c1:
                        #check distance is within h bond cutoff
                        if dist_check(coords[:,2:][p],coords[:,2:][q],latt) == 1 and p != q:
                            #determine h atoms bonded to donor atom
                            m0, = mols[np.where(mols[:,1] == coords[:,0][p])].astype(int)
                            #check angle is within h bond cutoff
                            if ang_check(coords[:,2:][coords[:,0] == m0[0]][0], coords[:,2:][p], coords[:,2:][q], latt) == 1 or ang_check(coords[:,2:][coords[:,0] == m0[2]][0], coords[:,2:][p], coords[:,2:][q], latt) == 1:
                                #count as hydrogen bond, store info in list to be returned by function
                                h_bonds.append(np.array([coords[:,0][p],coords[:,0][q]]))
                                #h_bonds = np.append(h_bonds,[coords[:,0][p],coords[:,0][q],ix[n]],axis=1)
                                
    return np.array(h_bonds)



