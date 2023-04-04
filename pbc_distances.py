#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 14:23:27 2022

@author: emmaarmstrong
"""
### FUNCTION TO CALCULATE DISTANCES WITHIN PERIODIC BOUNDARY CONDITIONS

import numpy as np

def distance(x0, x1, dimensions): #calculates distances between atoms taking into account periodic boundary conditions
    delta = np.abs(x0 - x1) 
    delta = np.where(delta > 0.5 * dimensions, delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

def gen_dist(r0,r1,h): #calculates distances between atoms across p.b.c. for non-orthogonl boxes
    h = h.T #converts to correct form required for calculation
    #r01 = r0 - r1
    s0 = np.dot(np.linalg.inv(h),r0)
    s1 = np.dot(np.linalg.inv(h),r1)
    s01 = s0 - s1
    s01 = s01 - np.rint(s01)
    r01 = np.dot(h,s01)
    return np.sqrt((r01 ** 2).sum(axis=-1))

def gen_angle(r0,r1,r2,h): #r0 MUST be central point for angle e.g. Ca in C-Ca-C
    h = h.T
    s0 = np.dot(np.linalg.inv(h),r0)
    s1 = np.dot(np.linalg.inv(h),r1)                                                                                          
    s2 = np.dot(np.linalg.inv(h),r2)
    s01 = s0 - s1
    s02 = s0 - s2
    s01 = s01 - np.rint(s01)
    s02 = s02 - np.rint(s02)
    r01 = np.dot(h,s01)
    r02 = np.dot(h,s02)
    v01 = r01 / np.linalg.norm(r01)
    v02 = r02 / np.linalg.norm(r02)
    return np.arccos(np.dot(v01,v02))

    