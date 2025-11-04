import numpy as np
import cv2
from skimage.registration import phase_cross_correlation
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool, cpu_count
from itertools import repeat, starmap
from functools import partial

def get_shifts_cv2_findTransformECC(reference_image, stack, n_threads=-1, **kwargs):
    """
    Compute translation shifts between a reference image and a stack of images using openCV's findTransformECC function
    Uses multi-threading to speed up computation, as this is more efficient than multiprocessing as cv2 releases the GIL
    Shifts are returned in image coordinates (x,y)
    Shifts are defined as to be directly applied to the stack to shift it to the reference image, e.g. via cv2.warpAffine
    args:
        reference_image: 2D array, reference image (h,w)
        stack: 3D array, stack of images (n_frames, h, w)
        n_threads: int, number of threads to use for parallel computation. If -1, use all cores.
        kwargs: additional keyword arguments to pass to findTransformECC
    returns:
        shifts: 2D array, translation shifts for each frame in the stack (n_frames, 2), in image coordinates (x,y)
    """

    # prepare arguments for cv2.findTransformECC
    kwargs["warpMatrix"] = np.eye(2,3, dtype=np.float32)       # initialize warp matrix as identity, as we don't expect large shifts
    kwargs["motionType"] = cv2.MOTION_TRANSLATION              # only use translation
    get_shifts = partial(cv2.findTransformECC, **kwargs)
    args = zip(stack.astype(np.float32), repeat(reference_image.astype(np.float32)))

    # compute shifts
    if n_threads == -1:
        n_threads = cpu_count()
    if n_threads == 1:
        affine_matrices = starmap(get_shifts, args)
    else:
        with Pool(n_threads) as pool:
            affine_matrices = pool.starmap(get_shifts, args)

    # convert affine matrices to shifts and return
    affine_matrices = np.stack([a[1] for a in affine_matrices])
    return affine_matrices_to_shifts(affine_matrices)

# import numba as nb
# @nb.jit(nopython=True)
# def phase_cc_numba(ref, mov, upsample_factor=20, normalization=None):
#     return phase_cross_correlation(ref, mov, upsample_factor=upsample_factor, normalization=normalization)

def get_shifts_phase_cross_corr(reference_image, stack, n_processes=-1, **kwargs):
    """
    Compute translation shifts between a reference image and a stack of images using scikit-image's phase_cross_correlation function.
    args:
        reference_image: 2D array, reference image (h,w)
        stack: 3D array, stack of images (n_frames, h, w)
        n_processes: int, number of processes to use for parallel computation. If -1, use all available cores.
        kwargs: additional keyword arguments to pass to phase_cross_correlation
    returns:
        shifts: 2D array, translation shifts for each frame in the stack (n_frames, 2), in image coordinates (x,y)
    """

    # prepare arguments for phase_cross_correlation
    get_shifts = partial(phase_cross_correlation, **kwargs)
    # get_shifts = partial(phase_cc_numba, **kwargs)
    args = zip(repeat(reference_image.astype(np.float64)), stack.astype(np.float64))

    # compute shifts
    if n_processes == -1:
        n_processes = cpu_count()
    with Pool(n_processes) as pool:
        shifts = pool.starmap(get_shifts, args)
    
    return np.stack([s[0] for s in shifts])[:,::-1]     # scikit-image returns shifts in (y,x) format, we want (x,y)

def get_shifts_cv2_phaseCorrelate(reference_image, stack, n_threads=-1, **kwargs):
    """
    Compute translation shifts between a reference image and a stack of images using openCV's phaseCorrelate function
    Uses multi-threading to speed up computation, as this is more efficient than multiprocessing as cv2 releases the GIL
    Shifts are returned in image coordinates (x,y)
    Shifts are defined as to be directly applied to the stack to shift it to the reference image, e.g. via cv2.warpAffine
    args:
        reference_image: 2D array, reference image (h,w)
        stack: 3D array, stack of images (n_frames, h, w)
        n_threads: int, number of threads to use for parallel computation. If -1, use all cores.
        kwargs: additional keyword arguments to pass to phaseCorrelate
    returns:
        shifts: 2D array, translation shifts for each frame in the stack (n_frames, 2), in image coordinates (x,y)
    """

    # determine if a smoothing kernel is provided
    if "sigmaXY" in kwargs:
        # apply smoothing to reference image
        reference_image = blur(reference_image, kwargs["sigmaXY"])
        # prepare arguments for _frame_shift_cv2_phaseCorrelate_with_smoothing
        get_shifts = partial(_frame_shift_cv2_phaseCorrelate_with_smoothing, **kwargs)
    else:
        # prepare arguments for cv2.phaseCorrelate
        get_shifts = partial(cv2.phaseCorrelate, **kwargs)

    args = zip(stack.astype(np.float32), repeat(reference_image.astype(np.float32)))

    # compute shifts
    if n_threads == -1:
        n_threads = cpu_count()
    if n_threads == 1:
        shifts = starmap(get_shifts, args)
    else:
        with ThreadPool(n_threads) as pool:
            shifts = pool.starmap(get_shifts, args)
    
    return np.stack([s[0] for s in shifts])

def blur(image, sigmaXY, truncate=5):
    """
    convenience wrapper for cv2.GaussianBlur
    """
    ksize = (int(sigmaXY[0] * truncate), int(sigmaXY[1] * truncate))
    return cv2.GaussianBlur(image, ksize, sigmaX=sigmaXY[0], sigmaY=sigmaXY[1])

def _frame_shift_cv2_phaseCorrelate_with_smoothing(mov, ref, sigmaXY, truncate=5, **kwargs):
    """
    helper function to compute shifts between a single frame and a reference image using cv2.phaseCorrelate with additional smoothing
    """
    mov = blur(mov, sigmaXY, truncate=truncate)
    return cv2.phaseCorrelate(mov, ref, **kwargs)

def shifts_to_affine_matrices(shifts):
    """
    converts translation shifts to affine transformation matrices (with no rotation), for use with cv2.warpAffine
    args:
        shifts: 2D array, translation shifts for each frame in the stack (n_frames, 2), in image coordinates (x,y)
    returns:
        affine_matrices: 3D array, affine transformation matrices for each frame (n_frames, 2, 3)
    """
    n_frames = shifts.shape[0]
    affine_matrices = np.zeros((n_frames, 2, 3), dtype=np.float32)
    affine_matrices[:,0,0] = np.float32(1)
    affine_matrices[:,1,1] = np.float32(1)
    affine_matrices[:,0,2] = shifts[:,0]
    affine_matrices[:,1,2] = shifts[:,1]
    return affine_matrices

def affine_matrices_to_shifts(affine_matrices):
    """
    converts affine transformation matrices to translation shifts
    args:
        affine_matrices: 3D array, affine transformation matrices for each frame (n_frames, 2, 3)
    returns:
        shifts: 2D array, translation shifts for each frame in the stack (n_frames, 2), in image coordinates (x,y)
    """
    return affine_matrices[:,:,2]

def test_phase_cross_corr(p, frames_fractions, n_processes, upsample_factor=20, normalization=None):
    import tifffile as tif
    stack = tif.imread(p)
    stack_blue = stack[::2]
    ref_blue = np.mean(stack_blue[:100], axis=0)

    frames_fractions = np.array(frames_fractions)
    n_frames = stack_blue.shape[0]
    frame_limits = (frames_fractions * n_frames).astype(int)
    #n_processes = [1, 4, 8, 16]

    from itertools import product
    from time import perf_counter
    # compare performance of phase_cross_correlation with different number of processes and frame limits
    # save results for plotting
    results = np.zeros((len(n_processes), len(frame_limits)))
    for i, n in enumerate(n_processes):
        for j, f in enumerate(frame_limits):
            t0 = perf_counter()
            shifts = get_shifts_phase_cross_corr(ref_blue, stack_blue[:f], n_processes=n, upsample_factor=upsample_factor, normalization=normalization)
            t = perf_counter()-t0
            print(f"n_processes = {n}, frame_limit = {f}, t = {t}", flush=True)
            results[i,j] = t
    
    # plot results, one line per n_processes
    import matplotlib.pyplot as plt
    plt.figure()
    for i, n in enumerate(n_processes):
        plt.plot(frame_limits, results[i], label=f"n_processes = {n}")
    plt.xlabel("Number of frames")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

def test_cv2_phaseCorrelate(p, frames_fractions, n_threads, **kwargs):
    import tifffile as tif
    stack = tif.imread(p)
    stack_blue = stack[::2]
    ref_blue = np.mean(stack_blue[:100], axis=0)

    frames_fractions = np.array(frames_fractions)
    n_frames = stack_blue.shape[0]
    frame_limits = (frames_fractions * n_frames).astype(int)
    #n_processes = [1, 4, 8, 16]

    from itertools import product
    from time import perf_counter
    # compare performance of phase_cross_correlation with different number of processes and frame limits
    # save results for plotting
    results = np.zeros((len(n_threads), len(frame_limits)))
    for i, n in enumerate(n_threads):
        for j, f in enumerate(frame_limits):
            t0 = perf_counter()
            shifts = get_shifts_cv2_phaseCorrelate(ref_blue, stack_blue[:f], n_threads=n, **kwargs)
            t = perf_counter()-t0
            print(f"n_threads = {n}, frame_limit = {f}, t = {t}", flush=True)
            results[i,j] = t
    
    # plot results, one line per n_processes
    import matplotlib.pyplot as plt
    plt.figure()
    for i, n in enumerate(n_threads):
        plt.plot(frame_limits, results[i], label=f"n_threads = {n}")
    plt.xlabel("Number of frames")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

def test_phase_cross_corr_2(p, upsample_factor=[1, 10, 20], frames_fractions=.2, n_processes=2, normalization=None):
    import tifffile as tif
    stack = tif.imread(p)
    stack_blue = stack[::2]
    ref_blue = np.mean(stack_blue[:100], axis=0, dtype=np.float32)

    frames_fractions = np.array(frames_fractions)
    n_frames = stack_blue.shape[0]
    frame_limits = (frames_fractions * n_frames).astype(int)
    #n_processes = [1, 4, 8, 16]

    from itertools import product
    from time import perf_counter
    # compare performance of phase_cross_correlation with different number of processes and frame limits
    # save results for plotting
    results = np.zeros((len(n_processes), len(frame_limits)))
    for i, n in enumerate(n_processes):
        for j, f in enumerate(frame_limits):
            t0 = perf_counter()
            shifts = get_shifts_phase_cross_corr(ref_blue, stack_blue[:f], n_processes=n, upsample_factor=upsample_factor, normalization=normalization)
            t = perf_counter()-t0
            print(f"n_processes = {n}, frame_limit = {f}, t = {t}", flush=True)
            results[i,j] = t
    
    # plot results, one line per n_processes
    import matplotlib.pyplot as plt
    plt.figure()
    for i, n in enumerate(n_processes):
        plt.plot(frame_limits, results[i], '-o', label=f"n_processes = {n}")
    plt.xlabel("Number of frames")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

def test_find_transform(p, frames_fractions, n_threads, criteria, **kwargs):
    import tifffile as tif
    stack = tif.imread(p)
    stack_blue = stack[::2]
    ref_blue = np.mean(stack_blue[:100], axis=0, dtype=np.float32)

    frames_fractions = np.array(frames_fractions)
    n_frames = stack_blue.shape[0]
    frame_limits = (frames_fractions * n_frames).astype(int)
    #n_processes = [1, 4, 8, 16]

    from itertools import product
    from time import perf_counter
    # compare performance of phase_cross_correlation with different number of processes and frame limits
    # save results for plotting
    results = np.zeros((len(n_threads), len(frame_limits)))
    for i, n in enumerate(n_threads):
        for j, f in enumerate(frame_limits):
            t0 = perf_counter()
            shifts = get_shifts_cv2_findTransformECC(ref_blue, stack_blue[:f], n_threads=n, **kwargs)
            t = perf_counter()-t0
            print(f"n_threads = {n}, frame_limit = {f}, t = {t}", flush=True)
            results[i,j] = t
    
    # plot results, one line per n_processes
    import matplotlib.pyplot as plt
    plt.figure()
    for i, n in enumerate(n_threads):
        plt.plot(frame_limits, results[i], '-o', label=f"n_threads = {n}")
    plt.xlabel("Number of frames")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    p = r"C:\Users\mpanze.UZH\Desktop\M064_2024-06-19_1_img.tif"

    # criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1)
    # test_find_transform(p, [.2], [1, 4, 8, 16], criteria)
    test_cv2_phaseCorrelate(p, [.2, .4, .6, .8, 1], [1, 4, 8, 16], sigmaXY=(5,5))

    # import cv2
    # #M = np.float32([[1,0,0.5],[0,1,-4.5]])
    # import matplotlib.pyplot as plt
    # #ref = stack[0]
    # #mov = cv2.warpAffine(stack[1], M, (512,512))
    # shifts, _, _ = phase_cross_correlation(ref, mov, upsample_factor=20, normalization=None)
    # cc, mcc = cv2.findTransformECC(ref.astype(np.float32), mov.astype(np.float32), np.eye(2,3, dtype=np.float32), cv2.MOTION_TRANSLATION, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01))
    # print(cc, mcc)
    # print(shifts)
    # M2 = np.float32([[1,0,shifts[1]],[0,1,shifts[0]]])
    # mov2 = cv2.warpAffine(mov, M2, (512,512))
    # f, ax = plt.subplots(1,3)
    # ax[0].imshow(ref, cmap="gray")
    # ax[1].imshow(mov, cmap="gray")
    # ax[2].imshow(mov2, cmap="gray")
    # plt.show()
