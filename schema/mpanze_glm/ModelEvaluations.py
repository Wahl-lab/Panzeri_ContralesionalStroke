# connect to the database
import login
login.connect()
import datajoint as dj

# import database dependencies
from schema.mpanze_glm.RidgeModel import RidgeModel
from schema.mpanze_glm.Regressor import Regressor
from schema.mpanze_glm.RidgeModelFit2 import RidgeModelFit2
from schema.mpanze_widefield_refactor import mpanze_widefield_refactor as wf
from schema.mpanze_paw_tracking_refactor import mpanze_paw_tracking_refactor as pt
from util.regression.SVDRidgeCV import SVDRidgeCV
from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp
from util.allen_utils import load_allen

# other dependencies
import numpy as np
import cv2

# instantiate the schema
schema = dj.schema('mpanze_glm', locals(), create_tables=True)

@schema
class BehavioralEvent(dj.Lookup):
    definition = """ # defines behavioral events for alignments
    event_id            : int                     # unique identifier for the event
    ---
    event_name          : varchar(64)             # name of the event
    """
    contents = [
        dict(event_id=0, event_name="Rewarded grasp onsets"),
        dict(event_id=1, event_name="Missed grasp onsets"),
        dict(event_id=2, event_name="Other task limb onsets"),
        dict(event_id=3, event_name="Support limb onsets"),
        dict(event_id=4, event_name="Reward valve opening onsets"),
        dict(event_id=5, event_name="Auditory cue onsets"),
    ]
    @staticmethod
    def load_event_timestamps(key):
        event_name = (BehavioralEvent & key).fetch1("event_name")
        if event_name == "Rewarded grasp onsets":
            return(
                pt.EpochClassification.Epoch.proj("epoch_class")
                * pt.MovementSegmentation.Epoch.proj("start_time")
                & dict(**key, epoch_class="rewarded")
            ).fetch("start_time")
        elif event_name == "Missed grasp onsets":
            return (
                pt.EpochClassification.Epoch.proj("epoch_class")
                * pt.MovementSegmentation.Epoch.proj("start_time")
                & dict(**key, epoch_class="miss")
            ).fetch("start_time")
        elif event_name == "Other task limb onsets":
            return (
                pt.EpochClassification.Epoch.proj("epoch_class")
                * pt.MovementSegmentation.Epoch.proj("start_time")
                & dict(**key, epoch_class="other")
            ).fetch("start_time")
        elif event_name == "Support limb onsets":
            return (
                pt.MovementSegmentation.Epoch.proj("start_time")
                * pt.MovementSegmentation.Hand
                & dict(**key, side='contra')
            ).fetch("start_time")
        elif event_name == "Reward valve opening onsets":
            return (
            exp.JoystickExperiment.Trials
            & key
            & "successful=1 OR autoreward=1"
        ).fetch("t_servo_out")
        elif event_name == "Auditory cue onsets":
            return (
                exp.JoystickExperiment.Trials
                & key
            ).fetch("t_cue")
        else:
            raise ValueError(f"Unknown event type: {event_name}")
        
@schema
class SegmentedPredictions(dj.Computed):
    definition = """ # allen atlas segmentation of model predictions
    -> RidgeModelFit2
    -> wf.RescaledAllenRegistration2
    -> exp.Handedness
    """
    class Subset(dj.Part):
        definition = """ # subset of predictions (incl. total and intercept)
        -> SegmentedPredictions
        subset_name        : varchar(64)            # name of the subset
        ---
        """
        def load_rois(self, roi_list):
            key = self.fetch1("KEY")
            roi_dffs, roi_ids = (
                SegmentedPredictions.ROI & key & [f'roi_id="{r}"' for r in roi_list]
            ).fetch("dff", "roi_id", order_by="roi_id")
            return roi_dffs, roi_ids
    
    class ROI(dj.Part):
        definition = """ # allen atlas rois
        -> SegmentedPredictions.Subset
        roi_id           : varchar(64)             # unique identifier for the ROI
        ---
        roi_name         : varchar(64)             # name of the ROI
        hemisphere       : enum('ipsi', 'contra')  # hemisphere of the ROI, relative to task limb
        dff              : longblob                # predicted dff signal for the roi (n_frames,)
        n_pixels         : int                     # number of valid pixels in the ROI
        """

    def make(self, key):
        # generate the subsets based on the model
        model_name = (RidgeModel & key).fetch1("model_name")
        if model_name == "FullModel_8" or model_name == "FullModel_0":
            subset_dict = {
                "task_limb_onsets": ["TaskLimbOnsets"],
                "support_limb_onsets": ["SupportLimbOnsets"],
                "valve_opening_events": ["ValveOpeningEvents"],
                "cue_events": ["CueEvents"],
                "task": ["CueEvents", "ValveOpeningEvents"],
            }
        else:
            raise NotImplementedError(f"Model {model_name} not implemented for event-aligned predictions")
        
        # fetch predictions
        predictions = (RidgeModelFit2 & key).load_predictions(assignment_dict=subset_dict, as_stack=False, include_true=False)

        # load info for registration
        allen_matrix = (wf.RescaledAllenRegistration2 & key).fetch1("allen_matrix_rescaled")
        # invert allen registration matrix
        # for computational efficiency, we warp the masks instead of the full stacks
        M = cv2.invertAffineTransform(allen_matrix)
        h, w = (wf.ImageProcessingParameters2 & key).fetch_hw()
        u = (wf.ImageProcessing2.SpatialComponents & key).fetch1("u").astype(np.float32)
        handedness = (exp.Handedness & key).fetch1("handedness")
        registered_mask = (wf.ImageProcessing2.RegisteredMask & key).fetch1("mask")
        n_frames = predictions["total"].shape[0]   # predictions are (n_frames, n_components)

        # load allen rois
        masks, area_names, edges, mask_total, bregma = load_allen((h,w))

        # register total mask to the imaging session
        mask_total = cv2.warpAffine(mask_total, M, (w, h))
        mask_valid = (registered_mask > 127) & (mask_total > 0)

        # iterate over the subsets
        entries_subsets = []
        entries_rois = []
        for subset_name, svt in predictions.items():
            # create the subset entry
            entry_subset = dict(**key, subset_name=subset_name)
            entries_subsets.append(entry_subset)

            # project to pixel space
            stack = (u @ svt.T).T.reshape(-1, h, w)  # (n_frames, h, w)

            # iterate over the rois
            for mask, area_name in zip(masks, area_names):
                name_split = area_name.split("_")
                # register the mask to the imaging session
                mask_roi = cv2.warpAffine(mask, M, (w, h))
                mask_roi_valid = mask_valid & (mask_roi > 0)
                n_pixels = np.sum(mask_roi_valid)
                if n_pixels > 0:
                    dff = np.nanmean(stack[:, mask_roi_valid], axis=1, dtype=np.float32)
                else:
                    dff = np.full(n_frames, np.nan, dtype=np.float32)

                hemisphere = "ipsi" if name_split[1]== handedness else "contra"

                entry_roi = dict(
                    **key,
                    subset_name=subset_name,
                    roi_id = f"{name_split[0]}_{hemisphere}",
                    roi_name = name_split[0],
                    hemisphere=hemisphere,
                    dff=dff,
                    n_pixels=n_pixels
                )
                entries_rois.append(entry_roi)
            del stack  # free memory
        
        # insert the entries
        self.insert1(key)
        self.Subset.insert(entries_subsets)
        self.ROI.insert(entries_rois)
