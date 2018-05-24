#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:57:09 2018

@author: bossema
"""
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp

def circle(pic_size, circle_nr):
    
    
    
    shape = np.array([pic_size,pic_size])
    xx = np.arange(0, shape[0]) - shape[0] // 2
    yy = np.arange(0, shape[1]) - shape[1] // 2
    
    all_circles = np.zeros((pic_size,pic_size))

    
    for i in range(circle_nr):
        parameters = [pic_size*(circle_nr-i)/(2*circle_nr), 0.02*pic_size]
        r0 = (parameters[0] - parameters[1])**2
        r1 = (parameters[0])**2
        vol = ((xx[:, None, None])**2 + (yy[None, :, None])**2)
        vol = np.array(((vol > r0) & (vol < r1)), dtype = 'float32') 
        vol = np.reshape(vol,(pic_size,pic_size))
        all_circles = all_circles + vol

    return all_circles



def wobbly_transform(image):

    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 10)
    src_rows = np.linspace(0, rows, 10)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    #dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0])) * 10

    rand_offset = 2.0*np.pi*np.random.rand()

    # TODO
    # The range of the random samples should depend on image size or be parameters.
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0]) + rand_offset) * 20 - (np.random.rand(src.shape[0]) - 0.5) * 10.0
    
    #dst_rows = src[:, 1] - (np.random.rand(src.shape[0]) - 0.5) * 15.0
    
    
    dst_cols = src[:, 0]

    #dst_rows *= 1.5
    #dst_rows -= 1.5 * 50
    dst = np.vstack([dst_cols, dst_rows]).T


    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0]
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))

    inv_src = tform.inverse(src)
    
    return out, src, inv_src
    

