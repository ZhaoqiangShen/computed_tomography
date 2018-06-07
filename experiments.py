import ipi_project as ipi
import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools


from cvxpy import *
from topolar import topolar

def plot_polar_angles(angles, dot_size = 60):
    #angles in degrees

    fig = plt.figure(figsize = (5,5))
    dot_size = 50
    ax = fig.add_subplot(111,projection = 'polar')
    ax.set_yticklabels([])
    ax.set_thetamin(0)
    ax.set_thetamax(180)    

    angles = angles/180*np.pi
    
    r = np.ones_like(angles)

    ax.scatter(angles, r, s= dot_size)


def sinogram_degredation(sinogram, projection_angles):

    # Intrested in how the reconstruct degrades as projections are removed.

    # In this experiment we start of with using all projection data.
    # Contigious chunks of projection angles are left out and when doing the reconstruction.

    # We want to find out if all projection contribute equally to the reconstrction.
    
    sirt_iter = 300
    full_reconstruction = ipi.reconstruct_image_sirt(projection_angles,
                                                     sinogram,
                                                     sirt_iter)
    
    prev = 0
    results = list()
    stepsize = 10
    for p in range(stepsize, sinogram.shape[0], stepsize):
        sub_sinogram = sinogram[prev:p, :]
        sub_projection_angles = projection_angles[prev:p]

        reconstruction = ipi.reconstruct_image_sirt(sub_projection_angles,
                                                    sub_sinogram,
                                                    sirt_iter)

        distance = ipi.dist(full_reconstruction, reconstruction)

        results.append((p, distance))
        #print("||True - T_{}|| =  {}".format(p, distance))                                     

        prev = p

    distances = [r[1] for r in results]
        
    plt.plot(distances, 'o')
    plt.show()


def one_angle_reconstruction(sinogram, angle_to_duplicate, verbose=False):

    all_same_sinogram = np.tile(sinogram[angle_to_duplicate], (sinogram.shape[0], 1))

    x = all_same_sinogram.shape[0]

    proj_angles = np.linspace(0, (x-1)*2.0*np.pi/x, x)

    n_iter = 500
    reconstructed = ipi.reconstruct_image_sirt(proj_angles, all_same_sinogram, n_iter)

    if verbose:
        plt.imshow(reconstructed, cmap='gray')
        plt.show()
    
    return reconstructed
    

def best_sparse_approximation(sinogram, projection_angles, sparse_size, trials):

    # Experiment to see how the reconstruction quality varies as we pick

    # 1. Create baseline using all projection angles.
    # 2. Take a random subset of projection angles of size 'sparse_size' and store the difference from the baseline.
    # 3. Repeat this 'trials' number of times.

    # In the end, plot the best and the worst reconstruction found.

    sirt_iter = 100
    full_reconstruction = ipi.reconstruct_image_sirt(projection_angles,
                                                     sinogram,
                                                     sirt_iter)

    best_reconstruction = None
    worst_reconstruction = None

    best_angles = np.zeros_like(projection_angles)
    worst_angles = np.zeros_like(projection_angles)

    min_delta = 10000
    max_delta = -1.0

    x, y = sinogram.shape
    results = list()
    for k in range(trials):
        sparse_sizeubset = np.random.choice(x, sparse_size, replace=False)
        reconstruction = ipi.reconstruct_image_sirt(projection_angles[sparse_sizeubset],
                                                    sinogram[sparse_sizeubset, :],
                                                    sirt_iter)
        distance = ipi.dist(full_reconstruction, reconstruction)

        if distance > max_delta:
            max_delta = distance
            worst_reconstruction = (sparse_sizeubset, reconstruction)
        elif distance < min_delta:
            min_delta = distance
            best_reconstruction = (sparse_sizeubset, reconstruction)
        results.append(distance)

    best_angles[best_reconstruction[0]] = 1.0
    worst_angles[worst_reconstruction[0]] = 1.0

    results.sort()

    plt.subplot(231)
    plt.imshow(best_reconstruction[1], cmap='gray')
    plt.subplot(232)
    plt.imshow(worst_reconstruction[1], cmap='gray')
    plt.subplot(233)
    plt.imshow(np.abs(best_reconstruction[1] - worst_reconstruction[1]), cmap='gray')
    #plt.subplot(234, projection='polar')
    #plot_polar_angles(best_angles)
    #plt.subplot(235, projection='polar')
    #plot_polar_angles(worst_angles)

    plt.subplot(236)
    plt.plot(results, 'o')

    plt.show()

    def make_angles(angles):
        length = angles.shape[0]
        return 180.0 * np.where(angles > 0)[0] / length 

    
    plot_polar_angles(make_angles(best_angles))
    plt.show()
    #plt.subplot(235, projection='polar')
    plot_polar_angles(make_angles(worst_angles))
    plt.show()


def random_angle_selection(current_angles, sinogram):

    # Randomized angle selection algorithm.
    # Pick new a random angle that is not is the set of known angles.
    
    max_index = sinogram.shape[0]

    while True:
        new_angle = np.random.randint(0, max_index)
        # Already picked, try another one
        if current_angles[new_angle] > 0:
            continue
        return new_angle

    
def gap_angle_selection(current_angles, sinogram):

    # Gap-angle selection algorithm from K.J Batenburg et al in
    # Dynamic angle slection in binary tomography

    # Based on the current angles, a new angle is selected as the midpoint
    # between the largest interval in the set of current angles.
    
    
    intervals = []

    x = sinogram.shape[0]

    projections = np.nonzero(current_angles)[0].tolist()

    # First and last points also creates one interval
    x_0 = projections[0]
    x_m = projections[-1]

    interval_width = x_0 + (x - x_m)

    # All other intervals are given by
    # [x0, x1), [x1, x2), ..., [x_{m-1}, x_m)

    intervals.append((x_m, x_0, interval_width))
        
    # Normal intervals
    prev = x_0
    for p in range(1, len(projections)):
        x_i = projections[p]
        interval_width = x_i - prev
        intervals.append((prev, x_i, interval_width))
        prev = x_i

    intervals.sort(key=lambda x: x[2], reverse=True)

    widest = intervals[0][2]
    last_index = 0
    # Find all intervals that has the widest width
    for i, interval in enumerate(intervals):
        if widest > interval[2]:
            break
        last_index = i

        interval_index = np.random.randint(0, last_index + 1)
        interval_to_split = intervals.pop(interval_index)

        if interval_to_split[0] < interval_to_split[1]:
            midpoint = (interval_to_split[0] + interval_to_split[1]) // 2
        else:
            lower = interval_to_split[1]
            upper = interval_to_split[0]
            half_width = (lower + x - upper) // 2

            # Modulus to wrap over negative numbers.
            midpoint = (lower - half_width) % x

    return midpoint


def bootstrapped_angleselection(current_angles, sinogram):

    if np.count_nonzero(current_angles) <= 3:
        return gap_angle_selection(current_angles, sinogram)


    sirt_iter = 100

    # Remove one angle and check how much the reconstruction degrades

    angle_index = np.nonzero(current_angles)[0].tolist()

    elements = len(angle_index)

    results = []

    x, y = sinogram.shape
    
    all_angles = np.linspace(0, (x-1)*np.pi/x, x)

    all_angles_reconstruction = ipi.reconstruct_image_sirt(all_angles[angle_index],
                                                           sinogram[angle_index, :],
                                                           sirt_iter)

    for drop_one_subset in itertools.combinations(angle_index, elements - 1):
        drop_one_subset = list(drop_one_subset)
        drop_one_reconstruction = ipi.reconstruct_image_sirt(all_angles[drop_one_subset],
                                                             sinogram[drop_one_subset, :],
                                                             sirt_iter)

        distance = ipi.dist(all_angles_reconstruction, drop_one_reconstruction)

        results.append((distance, drop_one_subset))

    results.sort(key=lambda x: x[0])
        
    print("min = {}, max = {}, rel {}".
          format(results[0][0], results[-1][0], results[0][0] / results[-1][0]))

    # TODO return something proper
    return gap_angle_selection(current_angles, sinogram)

    
def eval_angle_selection_alg(sinogram, all_angles, max_angles, select_angle):

    # Create one baseline using all projections
    sirt_iter = 100
    full_reconstruction = ipi.reconstruct_image_sirt(all_angles,
                                                     sinogram,
                                                     sirt_iter)


    # Start with one random projections
    current_angles = np.zeros_like(all_angles, dtype=np.int32)
    results = []

    current_angles[np.random.randint(0, all_angles.shape[0])] = 1
    
    index = 0
    while np.count_nonzero(current_angles) < max_angles:
        next_angle = select_angle(current_angles, sinogram)
        current_angles[next_angle] = 1

        angle_index = current_angles > 0

        reconstruction = ipi.reconstruct_image_sirt(all_angles[angle_index],
                                                    sinogram[angle_index, :],
                                                    sirt_iter)
        distance = ipi.dist(full_reconstruction, reconstruction)

        results.append(distance)

    return results


def angle_selection_experiment(sinogram, all_angles, max_angles, iterations):

    results = collections.defaultdict(list)

    methods = [(gap_angle_selection, 'gap_angle_selection', 'r'),
               (random_angle_selection, 'random_angle_selection', 'b')]
    plt.figure(figsize = (10,10))
    for angle_selection in methods:
        for i in range(iterations):
            result = eval_angle_selection_alg(sinogram, all_angles, max_angles=max_angles,
                                              select_angle=angle_selection[0])

            results[angle_selection[1]].append(result)


    plot_handles = dict()
    for angle_selection in methods:
        for i, result in enumerate(results[angle_selection[1]]):
            if i == 0:
                plt.plot(result, angle_selection[2], label=angle_selection[1])
            else:
                plt.plot(result, angle_selection[2])

    ax = plt.gca()
    ax.set_xlabel('Number of projections')
    ax.set_ylabel('Reconstruction error (MSE)')
    plt.legend()
    
    plt.show()

    
def angle_subset_reconstruction(sinogram, sirt_iter, proj_angles):
    # input the projection angles in degrees
    n_angles = len(proj_angles)
    scan_width = sinogram.shape[1]

    sinogram_subset = np.zeros((n_angles, scan_width))
    for i in range(n_angles):
        # or should we subtract 1?
        sinogram_subset[i, :] = sinogram[int(proj_angles[i]/360*1800), :]

    proj_angles = proj_angles/180*np.pi

    plt.imshow(sinogram_subset, cmap='gray')
    plt.show()
    return ipi.reconstruct_image_sirt(proj_angles, sinogram_subset, sirt_iter)


def tv_inpaint_sinogram(sinogram, proj_angles, drop_rate):


    # Start with all sinogram. Randomly drop say 90% of projections, i.e drop_rate = 0.9
    # Use Total-Variation inpainting to fill in the missing rows in the sinograms.
    # Use the inpainted sinogram and do a reconstrction.


    x, y = sinogram.shape



    known = np.zeros_like(sinogram)
    known_projections = np.random.choice(x, round((1.0 - drop_rate) * x), replace=False)


    to_keep = round((1.0 - drop_rate) * x)

    known_projections = np.linspace(0, x - 0.5, to_keep).astype(int)
    
    known[known_projections] = sinogram[known_projections]

    plt.subplot(121)
    plt.imshow(known, cmap='gray')



    restricted_reconstruction = ipi.reconstruct_image_sirt(proj_angles[known_projections],
                                                           sinogram[known_projections],
                                                           n_iter=500)

    plt.subplot(122)
    plt.imshow(restricted_reconstruction, cmap='gray')
    
    plt.show()
    
    # Set up TV-inpainting optimization problem
    U = Variable(x, y)
    obj = Minimize(tv(U))
    constraints = [mul_elemwise(known, U) == mul_elemwise(known, sinogram)]
    prob = Problem(obj, constraints)
    # Use SCS to solve the problem.
    prob.solve(verbose=True, solver=SCS, max_iters=2000)

    np.save("U_value.npy", U.value)

    restored_singram = U.value

    plt.figure(figsize = (13,13))
    plt.subplot(221)
    plt.imshow(known, cmap='gray')
    plt.subplot(222)
    plt.imshow(restricted_reconstruction, cmap='gray')
    plt.subplot(223)
    plt.imshow(restored_singram, cmap='gray')
    plt.subplot(224)
    
    tv_reconstruction = ipi.reconstruct_image_sirt(proj_angles, restored_singram,
                                                   n_iter=500)
    plt.imshow(tv_reconstruction, cmap='gray')
    plt.show()



def image2polar(sinogram, proj_angles):
    
    full_reconstruction = ipi.reconstruct_image_sirt(proj_angles, sinogram, 1000)
    # TODO find coordinate of smalled tree ring instead of hardcoding.
    pol, _ = topolar(full_reconstruction, x_center=499.0, y_center=429.0)

    plt.figure(figsize=(10,12))
    
    plt.imshow(pol[0:200], cmap='gray')
    plt.show()

