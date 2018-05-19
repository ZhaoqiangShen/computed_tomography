import ipi_project as ipi
import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools

def plot_polar_angles(projection_angles, max_range):
    ax = plt.gca()

    x = projection_angles.shape[0]
    
    angles = np.linspace(0, (x-1)*max_range/x, x)

    theta = angles[projection_angles > 0]
    r = np.ones_like(theta)

    ax.scatter(theta, r)


def sinogram_degredation(sinogram, projection_angles):

    """
    Experiment:

    We want to investigate how a reconstruction decays when projections are removed.


    We will remove projection in segments of size of n.

    """

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
        print("||True - T_{}|| =  {}".format(p, distance))                                     

        prev = p

    distances = [r[1] for r in results]
        
    plt.plot(distances, 'o')
    plt.show()


def best_sparse_approximation(sinogram, projection_angles, sparse_s):

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
    
    x,y = sinogram.shape
    trials = 250
    results = list()
    for k in range(trials):
        sparse_subset = np.random.choice(x, sparse_s, replace=False)
        reconstruction = ipi.reconstruct_image_sirt(projection_angles[sparse_subset],
                                                    sinogram[sparse_subset, :],
                                                    sirt_iter)
        distance = ipi.dist(full_reconstruction, reconstruction)

        if distance > max_delta:
            max_delta = distance
            worst_reconstruction = (sparse_subset, reconstruction)
            #worst_angles[sparse_subset] +=  1.0/(0.01 + distance)
        elif distance < min_delta:
            min_delta = distance
            best_reconstruction = (sparse_subset, reconstruction)
            #best_angles[sparse_subset] += 1.0/(0.01 + distance)
        
        
        print(k, distance)
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
    plt.subplot(234, projection='polar')
    plot_polar_angles(best_angles, np.pi)
    plt.subplot(235, projection='polar')
    plot_polar_angles(worst_angles, np.pi)

    plt.subplot(236)
    plt.plot(results, 'o')
    
    plt.show()


def random_angle_selection(current_angles, sinogram):

    max_index = sinogram.shape[0]

    while True:
        new_angle = np.random.randint(0, max_index)
        # Already picked, try another one
        if current_angles[new_angle] > 0:
            continue
        return new_angle

    
def dynamic_midpoint(current_angles, sinogram):
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
        return dynamic_midpoint(current_angles, sinogram)


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
    return dynamic_midpoint(current_angles, sinogram)

    
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

    methods = [(dynamic_midpoint, 'dynamic_midpoint', 'r'),
               (random_angle_selection, 'random_angle_selection', 'b')]
    plt.figure(figsize = (10,10))
    for angle_selection in methods:
        for i in range(iterations):
            result = eval_angle_selection_alg(sinogram, all_angles, max_angles=max_angles,
                                              select_angle=angle_selection[0])

            results[angle_selection[1]].append(result)

    for angle_selection in methods:
        for result in results[angle_selection[1]]:
            plt.plot(result, angle_selection[2])

    ax = plt.gca()
    ax.set_xlabel('Number of projections')
    ax.set_ylabel('Reconstruction error (MSE)')
    plt.show()
        


def run_exeperiment():
    sinogram = np.zeros((100,900))
    proj_angles = np.linspace(0, np.pi*99.0/100.0, 100)
    sinogram_degredation(sinogram, proj_angles)






