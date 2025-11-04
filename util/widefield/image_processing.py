import cv2
import numpy as np
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
from functools import partial
from itertools import repeat, starmap
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt
from sklearn.utils.extmath import randomized_svd

def spatial_process_frame(frame, mc_M=None, session_M=None, sigmaXY=None, new_res=None, truncate=5):
    """
    Apply spatial processing to single frame
    params:
        frame: ndarray of shape (h, w), input frame
        mc_M: ndarray of shape (2, 3), motion correction affine transform matrix, if None then no transformation is performed
        session_M: ndarray of shape (2, 3), session affine transform matrix, if None then no transformation is performed
        sigmaXY: tuple of ints (sigmaX, sigmaY), spatial smoothing sigma in image coordinates (x,y), if None then no smoothing is performed
        new_res: tuple of ints (h_new, w_new), new resolution, if None then no resizing is performed
        truncate: int, truncate gaussian kernel at this many standard deviations, default 5 (recommended)
    returns:
        frame_out: ndarray of shape (h_new, w_new), processed frame
    """
    h, w = frame.shape
    # apply affine transformations
    if mc_M is not None:
        frame_out = cv2.warpAffine(frame, mc_M, (w, h))
    else:
        frame_out = frame
    if session_M is not None:
        frame_out = cv2.warpAffine(frame_out, session_M, (frame.shape[1], frame.shape[0]))

    # apply spatial smoothing
    if sigmaXY is not None:
        ksize = (int(sigmaXY[0] * truncate), int(sigmaXY[1] * truncate))
        frame_out = cv2.GaussianBlur(frame_out, ksize=ksize, sigmaX=sigmaXY[0], sigmaY=sigmaXY[1])

    # resize
    if new_res is not None:
        frame_out = cv2.resize(frame_out, new_res, interpolation=cv2.INTER_AREA)

    return frame_out

def spatial_process_stack(stack, mc_M=None, session_M=None, sigmaXY=None, new_res=None, truncate=5, n_threads=-1):
    """
    Perform the following spatial processing steps on stack:
        - applies motion correction affine transformation
        - applies session registration affine transformation
        - gaussian smoothing in XY
        - resizing
    Uses multithreading to process stack in parallel
    params:
        stack: stack to process of shape (n_frames, h, w)
        mc_M: ndarray of shape (n_frames, 2, 3), motion correction affine transform matrix, if None then no transformation is performed
        session_M: ndarray of shape (2, 3), session affine transform matrix, if None then no transformation is performed
        sigmaXY: tuple of ints (sigmaX, sigmaY), spatial smoothing sigma in image coordinates (x,y), if None then no smoothing is performed
        new_res: tuple of ints (h_new, w_new), new resolution, if None then no resizing is performed
        truncate: int, truncate gaussian kernel at this many standard deviations, default 5 (recommended)
        n_threads: number of threads to use, -1 to use all available threads (recommended)
    returns:
        stack_out: ndarray of shape (n_frames, h_new, w_new), processed stack
    """
    # bind arguments to function
    if mc_M is None:
        mc_M = repeat(None)

    get_frame = partial(spatial_process_frame, session_M=session_M, sigmaXY=sigmaXY, new_res=new_res, truncate=truncate)
    args = zip(stack, mc_M)

    # process stack
    if n_threads == -1:
        n_threads = multiprocessing.cpu_count()
    with ThreadPool(n_threads) as pool:
        stack_out = pool.starmap(get_frame, args)
    
    # return as ndarray
    return np.stack(stack_out)

def temporal_process_pixel(pixel, fs=20., order=2, fcutoff=0.1, sigmaT=1, padlen=None):
    """
    Apply the following temporal processing steps to a single pixel:
        - high-pass filter
        - temporal smoothing with gaussian kernel
    params:
        pixel: ndarray of shape (n_frames,), input pixel time series
        fs: float, sampling frequency
        order: int, order of butterworth filter
        fcutoff: float, cutoff frequency for high-pass filter, if None then skip high-pass filtering
        sigmaT: float, temporal smoothing sigma in frames, if None then skip temporal smoothing
    returns:
        pixel_out: ndarray of shape (n_frames,), processed pixel time series
    """
    # high-pass filter
    if fcutoff is not None:
        sos = butter(order, fcutoff, btype='highpass', fs=fs, output='sos')
        pixel_out = sosfiltfilt(sos, pixel, padlen=padlen)
    else:
        pixel_out = pixel
    
    # temporal smoothing
    if sigmaT is not None:
        pixel_out = gaussian_filter1d(pixel_out, sigma=sigmaT)
    
    return pixel_out.astype(np.float32)

def temporal_process_stack(stack, fs=20., order=2, fcutoff=0.1, sigmaT=1, padlen=None, n_processes=-1):
    """
    Perform the following temporal processing steps on stack:
        - high-pass filter
        - temporal smoothing with gaussian kernel
    Uses multiprocessing to process stack in parallel
    params:
        stack: stack to process of shape (n_frames, h, w)
        fs: float, sampling frequency
        order: int, order of butterworth filter
        fcutoff: float, cutoff frequency for high-pass filter, if None then skip high-pass filtering
        sigmaT: float, temporal smoothing sigma in frames, if None then skip temporal smoothing
        n_processes: number of processes to use, -1 to use all available CPUs
    returns:
        stack_out: ndarray of shape (n_frames, h, w), processed stack
    """
    n_frames, h, w = stack.shape
    n_pixels = int(h) * int(w)
    # bind arguments to function
    get_pixel = partial(temporal_process_pixel, fs=fs, order=order, fcutoff=fcutoff, sigmaT=sigmaT, padlen=padlen)
    # reshape stack to (n_pixels, n_frames) for processing, subtract baseline to center around 0
    baseline = np.mean(stack, axis=0, dtype=np.float32)
    args = (stack-baseline).reshape((n_frames, n_pixels)).T

    # process stack
    if n_processes == -1:
        n_processes = multiprocessing.cpu_count()
    with Pool(n_processes) as pool:
        pixels_out = pool.map(get_pixel, args)
    
    # convert back to (n_frames, h, w) shape and add baseline back
    stack_out = np.stack(pixels_out).T.reshape((n_frames, h, w)) + baseline
    return stack_out

def compute_svd(stack, n_components, **kwargs):
    """
    Compute truncated SVD decomposition of stack and explained variance for each component
    params:
        stack: ndarray of shape (n_frames, h, w), input stack
        n_components: int, number of components to keep
        kwargs: additional keyword arguments to pass to randomized_svd
    returns:
        U: ndarray of shape (h*w, n_components), U matrix from SVD
        SVT: ndarray of shape (n_components, n_frames), SVT matrix from SVD
        var_expl: ndarray of shape (n_components,), variance explained by each component
    """
    frames, h, w = stack.shape
    # reshape stack to (n_frames, n_pixels) for SVD and fit
    U, S, VT = randomized_svd(
        stack.reshape((frames, h*w)).T,
        n_components=n_components,
        **kwargs
    )
    SVT = np.diag(S) @ VT         # has shape (n_components, n_frames)
    # compute explained variance for each component
    var_total = np.var(stack.reshape((frames, h*w)), axis=0).sum()            # total variance across features
    var_expl = np.var(SVT, axis=1)/var_total     # compute explained variance for each component
    return U, SVT, var_expl

def remove_background_pixel(f_uv, f_blue, svt_background):
    """
    Regress out the background and hemodynamics from a single pixel time series
    params:
        f_uv: 1D array, pixel time series from UV channel
        f_blue: 1D array, pixel time series from Blue channel
        svt_background: 2D array, SVT matrix of background components (shape: n_components x n_frames)
    returns:
        residuals: 1D array, residuals after regressing out background and hemodynamics
    """
    A = np.vstack([f_uv, svt_background, np.ones(len(f_uv))]).T
    coeffs = np.linalg.lstsq(A, f_blue, rcond=None)[0]
    f_fit = coeffs @ A.T
    return f_blue - f_fit

def remove_background_stack(stack_uv, stack_blue, svt_background):
    """
    Regress out the background and hemodynamics from a widefield stack
    params:
        stack_uv: 3D array of shape (n_frames, h, w), UV channel stack
        stack_blue: 3D array of shape (n_frames, h, w), Blue channel stack
        svt_background: 2D array of shape (n_components, n_frames), SVT matrix of background components
    returns:
        y_residuals: 3D array of shape (n_frames, h, w), residuals after regressing out background and hemodynamics
    """
    n_frames, h, w = stack_uv.shape
    args = zip(stack_uv.reshape(n_frames, h*w).T, stack_blue.reshape(n_frames, h*w).T, repeat(svt_background))
    y_residuals = np.array(list(starmap(remove_background_pixel, args)))
    return y_residuals.T.reshape(n_frames, h, w)

def remove_hemo_pixel(f_uv, f_blue):
    A = np.vstack([f_uv, np.ones(len(f_uv))]).T
    coeffs = np.linalg.lstsq(A, f_blue, rcond=None)[0]
    f_fit = coeffs @ A.T
    return f_blue - f_fit

def remove_hemo_stack(stack_uv, stack_blue):
    n_frames, h, w = stack_uv.shape
    args = zip(stack_uv.reshape(n_frames, h*w).T, stack_blue.reshape(n_frames, h*w).T)
    y_residuals = np.array(list(starmap(remove_hemo_pixel, args)))
    return y_residuals.T.reshape(n_frames, h, w)

if __name__ == "__main__":
    # test spatial processing
    from schema.mpanze_widefield_refactor.WidefieldSession import WidefieldSession
    from mpanze_scripts.util.widefield.motion_correction import shifts_to_affine_matrices
    from schema.mpanze_widefield_refactor.MotionCorrection import MotionCorrection
    from schema.mpanze_widefield_refactor.registration.RegisteredSession import RegisteredSession
    import tifffile as tif

    # session
    session = dict(mouse_id=64, day='2024-06-19', wf_param_id=3)
    
    # load necessary data
    p_tif = (WidefieldSession & session).get_path()
    stack = tif.imread(p_tif, key=range(0,10000,2))
    shifts = (MotionCorrection.Shifts & dict(**session, channel='blue')).fetch1("shifts")[:5000]
    mc_M = shifts_to_affine_matrices(shifts)
    # session_M = (RegisteredSession & session).fetch1("affine_matrix")
    session_M = None
    new_res = (256,256)
    sigmaXY = (3,3)

    # process stack
    stack_out = spatial_process_stack(stack, mc_M=mc_M, session_M=session_M, sigmaXY=sigmaXY, new_res=new_res, n_threads=16)

    # test different number of threads
    # from time import perf_counter
    # for n_threads in [1, 2, 4, 8, 16]:
    #     t0 = perf_counter()
    #     stack_out = spatial_process_stack(stack, mc_M=mc_M, session_M=session_M, sigmaXY=sigmaXY, new_res=new_res, n_threads=n_threads)
    #     print(f"Time elapsed with {n_threads} threads: {perf_counter()-t0:.3f} seconds")

    # process temporally
    # stack_out_t = temporal_process_stack(stack_out, fs=20., fcutoff=0.1, sigmaT=1, n_threads=8)

    # test different number of threads
    from time import perf_counter
    for n_threads in [1, 2, 4, 8, 16]:
        t0 = perf_counter()
        stack_out_t = temporal_process_stack(stack_out, fs=20., fcutoff=0.1, sigmaT=1, n_processes=n_threads)
        print(f"Time elapsed with {n_threads} threads: {perf_counter()-t0:.3f} seconds")

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,3)
    # ax[0].imshow(stack[0], cmap='gray')
    # ax[0].set_title("Raw")
    # ax[1].imshow(stack_out[0], cmap='gray')
    # ax[1].set_title("Spatial")
    # ax[2].imshow(stack_out_t[0], cmap='gray')
    # ax[2].set_title("Temporal")
    
    # fig, ax = plt.subplots()
    # plt.plot(stack[:,256,256], label="Raw")
    # plt.plot(stack_out[:,128,128], label="Spatial")
    # plt.plot(stack_out_t[:,128,128], label="Temporal")
    # plt.show()