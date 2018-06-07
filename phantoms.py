#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 14:57:09 2018

@author: bossema
"""
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
import matplotlib.pyplot as plt




def circle(pic_size, circle_nr, off_center_x = 0,off_center_y = 0, wall_thickness = None):
    #off_center_x positive is move to the left, negative move to the right
    #off_center y positive is move upwards, negative is move downward
    #off_center maximum 10% of picture width
    
    if wall_thickness == None:
        wall_thickness = 0.02*pic_size
    
    
    shape = np.array([pic_size,pic_size])
    xx = np.arange(0+off_center_y, shape[0]+off_center_y) - shape[0] // 2
    yy = np.arange(0+off_center_x, shape[1]+off_center_x) - shape[1] // 2
    
    all_circles = np.zeros((pic_size,pic_size))

    
    for i in range(circle_nr):
        parameters = [pic_size*(circle_nr-i)/(2*circle_nr)-0.1*pic_size, wall_thickness]
        r0 = (parameters[0] - parameters[1])**2
        r1 = (parameters[0])**2
        vol = ((xx[:, None, None])**2 + (yy[None, :, None])**2)
        vol = np.array(((vol > r0) & (vol < r1)), dtype = 'float32') 
        vol = np.reshape(vol,(pic_size,pic_size))
        all_circles = all_circles + vol

    return all_circles



def wobbly_transform(image, amplitude=None):

    rows, cols = image.shape[0], image.shape[1]

    size = 15
    
    src_cols = np.linspace(0, cols, size)
    src_rows = np.linspace(0, rows, size)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    rand_offset = 2.0*np.pi*np.random.rand()

    if amplitude is None:
        amplitude = 0.2*src.shape[0]

    dst_rows = src[:,1]

    dst_rows = (src[:, 1] - amplitude*np.sin(np.linspace(0, 3 * np.pi, src.shape[0]) + rand_offset)
                - amplitude/2.0*(np.random.rand(src.shape[0]) - 0.5))

    

    # We don't want to warp the center, because that makes the toplar function more complicated.
    # I.e we want to avoid finding the center and just assume that it's always in the center.

    center = np.array([rows / 2, cols / 2])

    ignore_radii = np.linalg.norm(center)*0.2
    print(ignore_radii)
    
    for i in range(src[:, 1].shape[0]):
        if np.linalg.norm(src[i,:] - center) < 20.0:

            dst_rows[i] = src[i,1]

    dst_cols = src[:, 0]

    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0]
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))

    inv_src = tform.inverse(src)


    #fig, ax = plt.subplots()
    #ax.imshow(out)
    #ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
    #ax.axis((0, out_cols, out_rows, 0))
    #plt.show()
    
    return out, src, inv_src
    

