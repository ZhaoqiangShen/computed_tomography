# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

pic_size = 128
radius = 0.9*pic_size/2
#If we want the full neighbours we have to do this for radiusses 1 upto for example 
#0.9*pic_size

shape = np.array([pic_size,pic_size])
xx = np.arange(0, shape[0]) - shape[0] // 2
yy = np.arange(0, shape[1]) - shape[1] // 2
    
parameters = [radius, 1]
r0 = (parameters[0] - parameters[1])**2
r1 = (parameters[0])**2
vol = ((xx[:, None, None])**2 + (yy[None, :, None])**2)
vol = np.array(((vol > r0) & (vol < r1)), dtype = 'float32') 
C = np.reshape(vol,(pic_size,pic_size))
pylab.imshow(C)

nonzero = np.argwhere(C !=0)

neighbours = []
for i in range(len(nonzero)):
    target = nonzero[i]
    distances = np.sqrt((nonzero[:,0] - target[0]) ** 2 + (nonzero[:,1] - target[1]) ** 2)
    distances[i]= np.inf
    nearest_indices = np.where(distances == distances.min())
    nearest_nonzero = nonzero[nearest_indices]
    #Left upper quarter
    if 0<=target[0]<=radius and 0<=target[1]<=radius:
        for j in range(len(nearest_nonzero)):
            if nearest_nonzero[j][0]<target[0]:
                neighbours.append(nearest_nonzero[j])
            if nearest_nonzero[j][1]>target[1]:
                neighbours.append(nearest_nonzero[j])
    #left lower quarter
    if 0<=target[0]<=radius and target[1]>=radius:
        for j in range(len(nearest_nonzero)):
            if nearest_nonzero[j][0]>target[0]:
                neighbours.append(nearest_nonzero[j])
            if nearest_nonzero[j][1]>target[1]:
                neighbours.append(nearest_nonzero[j])
#    #right lower quarter
    if target[0]>=radius and target[1]>=radius:
        for j in range(len(nearest_nonzero)):
            if nearest_nonzero[j][0]>target[0]:
                neighbours.append(nearest_nonzero[j])
            if nearest_nonzero[j][1]<target[1]:
                neighbours.append(nearest_nonzero[j])
#    #right upper quarter
    if target[0]>=radius and target[1]<=radius:
        for j in range(len(nearest_nonzero)):
            if nearest_nonzero[j][0]<target[0]:
                neighbours.append(nearest_nonzero[j])
            if nearest_nonzero[j][1]<target[1]:
                neighbours.append(nearest_nonzero[j])
                
#When we have the neighbours for all the pixels in all the circles this
#would have to be converted in a gigantic matrix imagesize*imagesize that has a 
#-1 on the diagonal and a 1 for each pixelneighbour (total 2 in every row)
#We could then have to think about when this becomes negative (ie take absolute value of gradient)
