"""
Pipeline for preprocessing motion-corrected widefield data
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema.mpanze_widefield_refactor.Registration import RegisteredSession, MaskedSession, AllenRegistration, ReferenceSession
from schema.mpanze_widefield_refactor.MotionCorrection import MotionCorrection
from schema.mpanze_widefield_refactor.WidefieldSession import WidefieldSession
from schema.mpanze_widefield_refactor.Synchronisation import Synchronisation as wf_Synchronisation
from schema.mpanze_paw_tracking_refactor.PawRecording import Synchronisation as pt_Synchronisation
from schema.mpanze_paw_tracking_refactor.Segmentation import RestingMask
from schema.mpanze_exp_refactor.Grouping import Handedness
from schema.mpanze_paw_tracking_refactor import mpanze_paw_tracking_refactor as pt

# other dependencies
import numpy as np
import cv2
import warnings
from util.widefield import image_processing as ip

# instantiate the schema
schema = dj.schema('mpanze_widefield_refactor', locals(), create_tables=True)

@schema
class ImageProcessingParameters2(dj.Lookup):
    definition = """ # Parameters for ImageProcessing2 pipeline
    wf_param_id     : int           # unique id for the preprocessing parameters
    ---
    sigma_x         : int           # spatial sigma in x direction
    sigma_y         : int           # spatial sigma in y direction
    new_h           : int           # new height of the image
    new_w           : int           # new width of the image
    order           : int           # order of the temporal filter
    fcutoff         : float         # cutoff frequency for the temporal filter
    sigma_t         : float         # temporal sigma for the filter
    n_components_compression    : int           # number of components for final svd
    n_components_background     : int           # number of components used for background
    n_dilations     : int           # number of dilations for the mask
    dilation_size   : int           # size of the dilation kernel
    """
    contents = [
        dict(
            wf_param_id = 6,
            sigma_x = 5,
            sigma_y = 5,
            new_h = 128,
            new_w = 128,
            order = 2,
            fcutoff = 0.1,
            sigma_t = 1.,
            n_components_compression = 100,
            n_components_background = 1,
            n_dilations = 5,
            dilation_size = 5
        ),
    ]
    def fetch_hw(self):
        return self.fetch1("new_h", "new_w")

@schema
class ImageProcessing2(dj.Computed):
    definition = """ # Preprocess motion-corrected widefield data
    -> RegisteredSession
    -> MaskedSession
    -> MotionCorrection
    -> ImageProcessingParameters2
    ---
    """

    def load_components(self, svt_baseline=False):
        key = self.fetch1("KEY")
        u = (ImageProcessing2.SpatialComponents & key).fetch1("u")
        if svt_baseline:
            svt = (BaselineSubtraction2 & key).fetch1("svt_subtracted")
        else:
            svt = (ImageProcessing2.TemporalComponents & key).fetch1("svt")
        h, w = (ImageProcessingParameters2 & key).fetch_hw()
        return u, svt, h, w
    
    class SpatialComponents(dj.Part):
        definition = """ # Spatial components of SVD
        -> ImageProcessing2
        ---
        u               : longblob       # spatial components of SVD (n_pixels, n_components)
        """
    
    class TemporalComponents(dj.Part):
        definition = """ # Temporal components of SVD
        -> ImageProcessing2
        ---
        svt             : longblob       # temporal components of SVD (n_components, n_frames)
        """

    class RegisteredMask(dj.Part):
        definition = """ # mask registered to reference session
        -> ImageProcessing2
        ---
        mask                    : longblob       # registered mask
        mask_dilated            : longblob       # dilated mask
        """

    def make(self, key):
        # get processing parameters
        params = (ImageProcessingParameters2 & key).fetch1()
        mc_M_blue = (MotionCorrection.Shifts & dict(**key, channel='blue')).get_M()
        mc_M_uv = (MotionCorrection.Shifts & dict(**key, channel='uv')).get_M()
        mask = (MaskedSession & key).fetch1("mask_session")
        session_M = (RegisteredSession & key).fetch1("affine_matrix")
        fps = (WidefieldSession & key).fetch1("approx_fps")

        # load the data
        stack_raw = (WidefieldSession & key).get_stack()
        raw_blue = stack_raw[::2]
        raw_uv = stack_raw[1::2]

        # apply spatial processing
        sigmaXY = (params["sigma_x"], params["sigma_y"])
        new_res = (params["new_h"], params["new_w"])
        spatial_blue = ip.spatial_process_stack(raw_blue, mc_M_blue, session_M, sigmaXY=sigmaXY, new_res=new_res).astype(np.float32)
        del raw_blue
        spatial_uv = ip.spatial_process_stack(raw_uv, mc_M_uv, session_M, sigmaXY=sigmaXY, new_res=new_res).astype(np.float32)
        del raw_uv

        # apply temporal processing
        fcutoff, order, sigma_t = params["fcutoff"], params["order"], params["sigma_t"]
        temporal_blue = ip.temporal_process_stack(spatial_blue, fs=fps, order=order, fcutoff=fcutoff, sigmaT=sigma_t)
        del spatial_blue
        temporal_uv = ip.temporal_process_stack(spatial_uv, fs=fps, order=order, fcutoff=fcutoff, sigmaT=sigma_t)
        del spatial_uv

        # register the mask
        mask = cv2.warpAffine(mask, session_M, (mask.shape[1], mask.shape[0]))
        mask = cv2.resize(mask, new_res, interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255
        # dilate the mask
        kernel = np.ones((params["dilation_size"], params["dilation_size"]), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=params["n_dilations"])
        mask_dilated = (mask_dilated > 127).astype(np.uint8) * 255

        # remove background and hemodynamics
        bg = np.mean(temporal_blue[:, mask==0], axis=1)
        stack_corr = ip.remove_background_stack(temporal_uv, temporal_blue, bg)

        # compute dff
        F0 = temporal_blue.mean(axis=0, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dff_corr = stack_corr / F0
        np.nan_to_num(dff_corr, copy=False, nan=0, posinf=0, neginf=0)
        dff_corr[:, mask_dilated == 0] = 0  # mask the data

        # compress the data
        u, svt, var = ip.compute_svd(dff_corr, n_components=params["n_components_compression"])

        # save the data
        self.insert1(key)
        self.SpatialComponents.insert1(dict(**key, u=u))
        self.TemporalComponents.insert1(dict(**key, svt=svt))
        self.RegisteredMask.insert1(dict(**key, mask=mask, mask_dilated=mask_dilated))


@schema
class RescaledAllenRegistration2(dj.Computed):
    definition = """ # Allen registration matrix rescaled to match ImageProcessing downsampled size
    -> AllenRegistration
    -> ImageProcessingParameters2
    --- 
    allen_matrix_rescaled : longblob   # affine transformation matrix to atlas for downsampled images
    """

    def make(self, key):
        # Fetch original Allen matrix and image processing parameters
        allen_matrix = (AllenRegistration & key).fetch1('allen_matrix')
        params = (ImageProcessingParameters2 & key).fetch1()
        new_h, new_w = params["new_h"], params["new_w"]

        # Fetch original image size directly
        ref_key = (ReferenceSession.proj(day="day_ref", session_num="num_ref") & key).fetch1("KEY")
        orig_w, orig_h = (WidefieldSession & ref_key).fetch1('pixels_w', 'pixels_h')

        # Compute scaling factors
        s_h = new_h / orig_h
        s_w = new_w / orig_w

        # Copy the original matrix
        allen_matrix_rescaled = np.array(allen_matrix, dtype=np.float64)
        # Scale the translation components
        allen_matrix_rescaled[0, 2] *= s_w
        allen_matrix_rescaled[1, 2] *= s_h

        self.insert1({**key, "allen_matrix_rescaled": allen_matrix_rescaled})

@schema
class BaselineSubtraction2(dj.Computed):
    definition = """ # Baseline subtraction for widefield data
    -> ImageProcessing2
    -> wf_Synchronisation
    -> pt_Synchronisation
    -> RestingMask
    ---
    svt_subtracted : longblob   # temporal components after baseline subtraction (n_components, n_frames)
    baseline       : longblob   # baseline used for subtraction (h, w)
    baseline_mask  : longblob   # mask used for baseline computation (n_frames,)
    """
    def load_components(self):
        key = self.fetch1("KEY")
        u = (ImageProcessing2.SpatialComponents & key).fetch1("u")
        svt = (BaselineSubtraction2 & key).fetch1("svt_subtracted")
        h, w = (ImageProcessingParameters2 & key).fetch_hw()
        return u, svt, h, w

    def make(self, key):
        # fetch the required data
        u = (ImageProcessing2.SpatialComponents & key).fetch1("u")
        svt = (ImageProcessing2.TemporalComponents & key).fetch1("svt")
        mask_dilated = (ImageProcessing2.RegisteredMask & key).fetch1("mask_dilated")
        h, w = (ImageProcessingParameters2 & key).fetch_hw()
        t_wf = (wf_Synchronisation & key).fetch1("frame_timestamps_blue")
        t_L = (pt_Synchronisation.Hand & dict(**key, hand='L')).fetch1("frame_timestamps")
        t_R = (pt_Synchronisation.Hand & dict(**key, hand='R')).fetch1("frame_timestamps")
        mask_L = (RestingMask.Hand & dict(**key, hand='L')).fetch1("paw_still_mask")
        mask_R = (RestingMask.Hand & dict(**key, hand='R')).fetch1("paw_still_mask")

        # interpolate timestamps to match widefield data
        from scipy.interpolate import interp1d
        f_L = interp1d(t_L, mask_L, kind='nearest', fill_value="extrapolate", bounds_error=False)
        f_R = interp1d(t_R, mask_R, kind='nearest', fill_value="extrapolate", bounds_error=False)
        mask_L_interp = f_L(t_wf).astype(bool)
        mask_R_interp = f_R(t_wf).astype(bool)
        baseline_mask = mask_L_interp & mask_R_interp

        # compute baseline
        stack = (u @ svt).T.reshape(-1, h, w)
        baseline = np.median(stack[baseline_mask], axis=0).astype(np.float32)
        stack_subtracted = stack - baseline
        stack_subtracted[:, mask_dilated == 0] = 0  # re-apply mask
        np.nan_to_num(stack_subtracted, copy=False, nan=0, posinf=0, neginf=0)
        svt_subtracted = u.T @ stack_subtracted.reshape(-1,h*w).T

        # insert the results
        entry = dict(**key, svt_subtracted=svt_subtracted, baseline=baseline, baseline_mask=baseline_mask)
        self.insert1(entry)

@schema
class AllenSegmentation2(dj.Computed):
    definition = """ # segment widefield data using Allen atlas
    -> BaselineSubtraction2
    -> RescaledAllenRegistration2
    -> Handedness
    """
    def load_rois(self, roi_list):
        key = self.fetch1("KEY")
        roi_dffs, roi_ids = (
            AllenSegmentation2.ROI & key & [f'roi_id="{r}"' for r in roi_list]
        ).fetch("dff", "roi_id", order_by="roi_id")
        return roi_dffs, roi_ids


    class ROI(dj.Part):
        definition = """ # allen atlas rois
        -> AllenSegmentation2
        roi_id          : varchar(64)  # unique id for the roi
        ---
        roi_name        : varchar(64)  # name of the roi
        hemisphere      : enum('ipsi', 'contra')   # hemisphere of the roi, relative to task paw
        dff             : longblob       # dff signal for the roi (n_frames,)
        n_pixels        : int          # number of valid pixels in the roi
        """

    def make(self, key):
        from mpanze_scripts.util.allen_utils import load_allen
        # fetch the stack
        svt = (BaselineSubtraction2 & key).fetch1("svt_subtracted")
        u = (ImageProcessing2.SpatialComponents & key).fetch1("u")
        h, w = (ImageProcessingParameters2 & key).fetch_hw()
        stack = (u @ svt).T.reshape(-1, h, w).astype(np.float32)
        n_frames = stack.shape[0]
        registered_mask = (ImageProcessing2.RegisteredMask & key).fetch1("mask")

        # load allen registration matrix
        allen_matrix = (RescaledAllenRegistration2 & key).fetch1("allen_matrix_rescaled")
        M = cv2.invertAffineTransform(allen_matrix)

        # load allen rois
        masks, area_names, edges, mask_total, bregma = load_allen((h,w))

        # register total mask to the imaging session
        mask_total = cv2.warpAffine(mask_total, M, (w, h))
        mask_valid = (registered_mask == 255) & (mask_total > 0)

        # get handedness
        handedness = (Handedness & key).fetch1("handedness")

        # iterate over the rois
        entries_rois = []
        for mask, area_name in zip(masks, area_names):
            name_split = area_name.split("_")
            # register mask to imaging session
            mask_roi = cv2.warpAffine(mask, M, (w, h))
            mask_roi_valid = (mask_roi > 0) & mask_valid
            n_pixels = np.sum(mask_roi_valid)
            if n_pixels > 0:
                dff = np.nanmean(stack[:, mask_roi_valid], axis=1)
            else:
                dff = np.zeros(n_frames, dtype=np.float32) * np.nan

            hemisphere = "ipsi" if name_split[1] == handedness else "contra"

            entry_roi = dict(
                **key,
                roi_id = f"{name_split[0]}_{hemisphere}",
                roi_name = name_split[0],
                hemisphere = hemisphere,
                dff = dff,
                n_pixels = n_pixels
            )
            entries_rois.append(entry_roi)
        # insert the entries
        self.insert1(key)
        self.ROI.insert(entries_rois)
