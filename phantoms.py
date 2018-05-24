#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:57:09 2018

@author: bossema
"""
import numpy as np

def circle(pic_size):
    parameters = [pic_size/2, 0.02*pic_size]
    shape = np.array([pic_size,pic_size])
    xx = np.arange(0, shape[0]) - shape[0] // 2
    yy = np.arange(0, shape[1]) - shape[1] // 2
    
    r0 = (parameters[0] - parameters[1])**2
    r1 = (parameters[0])**2

    vol = ((xx[:, None, None])**2 + (yy[None, :, None])**2)
    vol = np.array(((vol > r0) & (vol < r1)), dtype = 'float32')  
    P = np.reshape(vol,(pic_size,pic_size))

    return P

