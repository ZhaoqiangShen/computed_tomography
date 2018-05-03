import numpy as np
import os
import glob
import skimage.io
import matplotlib.pyplot as plt
import re
import astra


def load_images(directory):
    images = glob.glob(directory + '/scan*.tif')
    # Make sure we process the files in the same order they where scanned.
    images.sort(key=lambda f: int(re.search(r'scan_([0-9]+).tif', f).group(1)))

    first_image = skimage.io.imread(images[0])
    x, y = first_image.shape
    sinogram = np.zeros((len(images), y))
    
    for i, imagefile in enumerate(images):
        image = skimage.io.imread(imagefile)
        sinogram[i] = image[x // 2]

    # Last scan angle == first scan angle, so we drop the last scan
    return sinogram[:-1], images


if __name__ == '__main__':

    scans_directory = '/mnt/datasets1/fgustafsson/cwi_ct_scan/wooden_block/'
    sinogram, images = load_images(scans_directory)

    plt.imshow(sinogram, cmap='gray')
    plt.show()
    
    # As a sanity check look at first scan at angle 0
    first_image = skimage.io.imread(images[0])
    plt.imshow(first_image, cmap='gray')
    plt.show()


    scanned_angles = sinogram.shape[0]
    scan_width = sinogram.shape[1]


    # Work in progres
    n_iter = 5
    vol_geom = astra.create_vol_geom(scan_width, scan_width)
    proj_angles = np.linspace(0, (scanned_angles -1.0)*2.0*np.pi / scanned_angles, scanned_angles)
    print(proj_angles[0])
    print(proj_angles[-1])
    proj_geom = astra.create_proj_geom('fanflat', 1.0, scan_width, proj_angles, 313.0, 498.0)
    proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)
    W = astra.OpTomo(proj_id)
    f_inv  = W.reconstruct('SIRT', sinogram, iterations=n_iter,extraOptions={'MinConstraint':0.0})
