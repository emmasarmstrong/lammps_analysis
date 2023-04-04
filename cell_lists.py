#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:53:03 2022

@author: emmaarmstrong
"""

###Cell lists and rdf testing

import numpy as np
import matplotlib.pyplot as plt
import time
from line_profiler import LineProfiler
from math import acos, sqrt, cos, sin

#function to divide and sort atoms into cells
def cell_lists(n_cells,r_cut,n_atoms,coords): #number of cells in x, y and z directions, maximum potential cutoff, number of atoms and array of atom coordinates
    #empty array for storing assigned atom indices
    cells = np.zeros(n_cells,dtype=object)
    
    for i in range(n_atoms):
        #assigning atoms to cells based on x, y, z coordinates
        c = (coords[i,2:]/r_cut).astype(int)
        if cells[c[0],c[1],c[2]] == 0:
            cells[c[0],c[1],c[2]] = [i]
     #if atoms already present in cell, append atom index to exisiting list
        else:
            cells[c[0],c[1],c[2]].append(i)
    #return mixed numpy array with integer 0 values for empty cells and list objects in filled cells containg atom indices to correspond with stored coordinate list
    return cells

#function to divide and sort atoms into cells
def rdf_cell_lists(n_cells,r_cut,n_atoms,coords,types): #number of cells in x, y and z directions, maximum potential cutoff, number of atoms and array of atom coordinates
    #empty array for storing assigned atom indices
    cells = np.zeros(n_cells,dtype=object)
    
    for i in range(n_atoms):
        #assigning atoms to cells based on x, y, z coordinates
        c = (coords[i]/r_cut).astype(int)
        if c[0] == n_cells[0]:
            c[0] = c[0] - 1
        if c[1] == n_cells[1]:
            c[1] = c[1] - 1
        if c[2] == n_cells[2]:
            c[2] = c[2] - 1
        if c[0] > n_cells[0]:
            c[0] = 0
        if c[1] > n_cells[1]:
            c[1] = 0
        if c[2] > n_cells[2]:
            c[2] = 0
        
        
        if cells[c[0],c[1],c[2]] == 0:
            cells[c[0],c[1],c[2]] = [i]
    #if atoms already present in cell, append atom index to exisiting list
        else:
            cells[c[0],c[1],c[2]].append(i)
    #return mixed numpy array with integer 0 values for empty cells and list objects in filled cells containg atom indices to correspond with stored coordinate list
    return cells

def neighbours(i,j,k,n_cells):
    ilo = n_cells[0] if i == 0 else 0
    ihi = -n_cells[0] if i == n_cells[0] - 1 else 0
    jlo = n_cells[1] if j == 0 else 0
    jhi = -n_cells[1] if j == n_cells[1] - 1 else 0
    klo = n_cells[2] if k == 0 else 0
    khi = -n_cells[2] if k == n_cells[2] - 1 else 0    
    x,y,z = [i-1+ilo,i,i+1+ihi],[j-1+jlo,j,j+1+jhi],[k-1+klo,k,k+1+khi]
    ind = [[a,b,c] for a in x for b in y for c in z]
    return ind


def cell_distances(n_cells,cells,coords):
    #identify filled cells
    rxyz = []
    ix,iy,iz = np.nonzero(cells)
    n_filled = len(ix)
    #loop through filled cells
    for n in range(n_filled):
        #assign indices
        c0 = cells[ix[n],iy[n],iz[n]]
        #identify neighbouring cells
        #loop through neighbours
        for ind in neighbours(ix[n], iy[n], iz[n], n_cells):
            c1 = cells[ind[0],ind[1],ind[2]]
            if c1 != 0:
                dist = [coords[q] - coords[p] for p in c0 for q in c1 if p != q]
                rxyz += dist
        cells[ix[n],iy[n],iz[n]] = 0
    return rxyz

def rdf_cells_old(n_cells,cells0,cells1,coords):
    rxyz = []
    ix,iy,iz = np.nonzero(cells0)
    n_filled = len(ix)
    for n in range(n_filled):
        c0 = cells0[ix[n],iy[n],iz[n]]
        for ind in neighbours(ix[n],iy[n],iz[n],n_cells):
            c1 = cells1[ind[0],ind[1],ind[2]]
            if c1 != 0:
                dist = [coords[q] - coords[p] for p in c0 for q in c1]
                rxyz += dist
    return rxyz

# def rdf_cells(n_cells,cells,coords):
#     rxyz = []
#     ix,iy,iz = np.nonzero(cells)
#     n_filled = len(ix)
#     for n in range(n_filled):
#         c0 = cells[ix[n],iy[n],iz[n]]
#         for ind in neighbours(ix[n],iy[n],iz[n],n_cells):
#             c1 = cells[ind[0],ind[1],ind[2]]
#             if c1 != 0:
#                 dist = [coords[q] - coords[p] for p in c0 for q in c1 if ]
    

def lammpstocell(lx,ly,lz,xy,xz,yz):
    a = lx
    b = sqrt(ly**2+xy**2)
    c = sqrt(lz**2+xz**2+yz**2)
    alpha = acos((xy*xz + ly*yz)/(b*c))
    beta = acos(xz/c)
    gamma = acos(xy/b)
    alat = [a, b*cos(gamma), c*cos(beta)]
    blat = [0.0, b*sin(gamma), c*(cos(alpha)-cos(gamma)*cos(beta))/sin(gamma)]
    clat = [0.0, 0.0, c*(1.0 - cos(beta)**2-sqrt((blat[2]/c)**2))]
    lat = (np.array([alat,blat,clat])).T
    return lat
    

def lammpstolatt(bound):
    latt = np.zeros([3,3])
    latt[0,0] = bound[0,1] - bound[0,0] - abs(bound[0,2]) - abs(bound[1,2])
    latt[1,1] = bound[1,1] - bound[1,0] - abs(bound[2,2])
    latt[2,2] = bound[2,1] - bound[2,0]
    latt[1,0] = bound[0,2]
    latt[2,0] = bound[1,2]
    latt[2,1] = bound[2,2]
    return latt

# start = time.time()

#trajectory file
filepath = '/Users/emmaarmstrong/Desktop/IFE/ordering/aragonite/001/300/'
f = open(filepath+'prod_traj.lmp','r').readlines()

#specify number of frames 
frames = 501 #same for all IFE calculations
r_cutoff = 3

#get number of atoms and box size from file
n_atoms = int(f[3])
# xlo_bound,xhi_bound,xy = [float(x) for x in f[5].split()]
# ylo_bound,yhi_bound,xz = [float(x) for x in f[6].split()]
# zlo,zhi,yz = [float(x) for x in f[7].split()]
bound = np.array([[float(x) for x in f[5].split()],[float(x) for x in f[6].split()],[float(x) for x in f[7].split()]])
# xlo = xlo_bound - min(0.0,xy,xz,xy+xz)
# xhi = xhi_bound - max(0.0,xy,xz,xy+xz)
# ylo = ylo_bound - min(0.0,yz)
# yhi = yhi_bound - max(0.0,yz)
# latt = lammpstocell(xhi-xlo, yhi-ylo, zhi-zlo, xy, xz, yz)

# latt = np.array([f[5].split(),f[6].split(),f[7].split()])
# latt = latt.astype(float)

latt = lammpstolatt(bound)

#atom data 
coords = np.zeros([frames,n_atoms,5])

for i in range(frames):
    for j in range(n_atoms):
        data = f[9+i*(9+n_atoms)+j].split()
        coords[i][j] = data[0:5]
        
del f,data

start = time.time()

#r_cut = np.array([r_cutoff/(xhi-xlo),r_cutoff/(yhi-ylo),r_cutoff/(zhi-zlo)])
r_cut = np.dot(np.linalg.inv(latt),np.array([r_cutoff,r_cutoff,r_cutoff]))

n_cells = np.ceil(1.0/r_cut).astype(int)



#cells = cell_lists(n_cells, r_cut, n_atoms, coords[0][:,2:])

# lp = LineProfiler()
# lp_wrapper = lp(cell_dist2)
# lp_wrapper(n_cells,cells,coords[0][:,2:])

#rxyz = cell_dist2(n_cells, cells, coords[0][:,2:])

# lp_wrapper = lp(cell_distances)
# lp_wrapper(n_cells,cells,coords[0][:,2:])


# lp_wrapper = lp(neighbours2)
# lp_wrapper(30,0,3,n_cells)

#lp.print_stats()plt.h

r = np.empty(0)

types = np.array([4,5]) #lammps type values for water
for i in range(1):
    cells = cell_lists(n_cells, r_cut, n_atoms, coords[i])
    # rxyz = rdf_cells(n_cells, o_cells, h_cells, coords[i,:,2:])
    # rxyz = rxyz - np.rint(rxyz)
    # print(max(rxyz[:,0]),max(rxyz[:,1]),max(rxyz[:,2]))
    # r_frame = np.tensordot(latt,rxyz,(1,1)).T
    # print(max(r_frame[:,0]),max(r_frame[:,1]),max(r_frame[:,2]))
    # cells = cell_lists(n_cells, r_cut, n_atoms, coords[i,:,2:])
    # rxyz = cell_distances(n_cells, cells, coords[i][:,2:])
    # rxyz = rxyz - np.rint(rxyz)
    # r_frame = np.tensordot(latt,rxyz,(1,1)).T
    # r_frame = np.sqrt(np.sum(r_frame**2,axis=1))
    # r = np.append(r,r_frame)
    #print(f'Frame {i}')

    
 
#account for pbc
# rxyz = rxyz - np.rint(rxyz)
# r =  np.sqrt((rxyz**2).sum(axis=1))

#atom types for rdf calculation
# types = [4,5]
# n0 = int(np.count_nonzero(coords[:,:,1] == types[0])/frames)
# n1 = int(np.count_nonzero(coords[:,:,1] == types[1])/frames)


# r = np.zeros([frames*n0*n1,3])
# for j in range(frames):
#     a0 = coords[j,:,2:][coords[j,:,1] == types[0]]
#     a1 = coords[j,:,2:][coords[j,:,1] == types[1]]
#     for i in range(n0):
#         d = a0[i] - a1
#         r[n1*i+n1*j:n1*(i+1)+n1*j] = d - np.rint(d)
#         #r[n1*i+n1*j:n1*(i+1)+n1*j] = np.sqrt((d ** 2).sum(axis=-1))

# r01 = np.zeros(frames*n0*n1)
# for i in range(frames*n0*n1):
#     r01[i] = np.sqrt((np.dot(latt,r[i]) ** 2).sum(axis=-1))
                

        
#dist = np.zeros([frames, int(n0*n1)])

# for i in range(frames):
#     a0 = coords[i,:,2:][coords[0,:,1] == types[0]]
#     a1 = coords[i,:,2:][coords[0,:,1] == types[1]]
#     for j in range(n0):
#         dist[i,j*n1:(j+1)*n1] = np.linalg.norm(a1[j]-a0)
        
elapsed = time.time()-start
print(elapsed)

