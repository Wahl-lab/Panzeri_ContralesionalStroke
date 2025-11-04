"""
@author: mpanze

Export data to NWB format for manuscript submission
"""

# database imports
from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp
from schema.mpanze_widefield_refactor import mpanze_widefield_refactor as wf
from schema.mpanze_paw_tracking_refactor import mpanze_paw_tracking_refactor as pt
from schema import common_mice as mice
# NWB imports
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.behavior import BehavioralTimeSeries
from pynwb.epoch import TimeIntervals
from pynwb.ophys import OpticalChannel, ImageSegmentation, RoiResponseSeries, DfOverF
from pynwb.base import TimeSeries
# other imports
from pathlib import Path
from tqdm import tqdm
import datetime
from uuid import uuid4
import numpy as np


### functions for NWB export ###
def export_subject(key):
    mouse_info = (mice.Mouse & key).fetch1()
    mouse_ID = f"M{mouse_info['mouse_id']:03d}"
    session_date = (exp.JoystickExperiment & key).fetch1("day")
    age = session_date - mouse_info["dob"]

    subject = Subject(
        subject_id = mouse_ID,
        age = f"P{age.days}D",
        age__reference = "birth",
        species = "Mus musculus",
        sex = mouse_info["sex"],
        genotype = "C57BL/6J-Tg(Thy1-GCaMP6f)GP5.17Dkim/J +/-",
    )
    return subject

def export_session_description(key):
    mouse_ID = f"M{key['mouse_id']:03d}"
    session_date = key["day"]
    rel_day = (exp.DaysFromStroke & key).fetch1("days_from_stroke")
    phase = (exp.ExperimentalPhase & key).fetch1("phase")

    map_phase = {'Expert': 'Pre-stroke', 'Early': 'Early post-stroke', 'Late': 'Late post-stroke'}

    description = {
        "subject_id": mouse_ID,
        "date": session_date.strftime("%Y-%m-%d"),
        "phase": map_phase[phase],
        "days relative to stroke surgery": int(rel_day),
    }

    return f"{description}"

def export_limb_DLC(key, side):
    # export deeplabcut tracking of limb data
    key_limb = (pt.PawRecording.Hand & dict(**key, side=side)).fetch1('KEY')
    # fetch tracked keypoints
    label_keys = (pt.DeepLabCut.Label & key_limb).fetch('KEY')
    timestamps = (pt.Synchronisation.Hand & key_limb).fetch1("frame_timestamps")
    # create BehavioralTimeSeries
    behavioral_timeseries = BehavioralTimeSeries(
        name = "TaskLimbDeepLabCut" if side=='ipsi' else "SupportLimbDeepLabCut",
    )
    # define conversion from label names to NWB time series names
    label_name_map = {
        "wrist": "Tracked wrist coordinates (x, y, p) from DeepLabCut",
        "elbow": "Tracked elbow coordinates (x, y, p) from DeepLabCut",
        "1_knuckle": "Tracked knuckle coordinates (x, y, p) of 1st finger from DeepLabCut",
        "2_knuckle": "Tracked knuckle coordinates (x, y, p) of 2nd finger from DeepLabCut",
        "3_knuckle": "Tracked knuckle coordinates (x, y, p) of 3rd finger from DeepLabCut",
        "4_knuckle": "Tracked knuckle coordinates (x, y, p) of 4th finger from DeepLabCut",
        "1_mid": "Tracked midpoint coordinates (x, y, p) of 1st finger from DeepLabCut",
        "2_mid": "Tracked midpoint coordinates (x, y, p) of 2nd finger from DeepLabCut",
        "3_mid": "Tracked midpoint coordinates (x, y, p) of 3rd finger from DeepLabCut",
        "4_mid": "Tracked midpoint coordinates (x, y, p) of 4th finger from DeepLabCut",
        "1_tip": "Tracked fingertip coordinates (x, y, p) of 1st finger from DeepLabCut",
        "2_tip": "Tracked fingertip coordinates (x, y, p) of 2nd finger from DeepLabCut",
        "3_tip": "Tracked fingertip coordinates (x, y, p) of 3rd finger from DeepLabCut",
        "4_tip": "Tracked fingertip coordinates (x, y, p) of 4th finger from DeepLabCut",
    }

    for label_key in label_keys:
        x, y, p = (pt.DeepLabCut.Label & label_key).fetch1('x', 'y', 'p')
        coords = np.stack([x,y,p]).T
        # add data to BehavioralTimeSeries
        behavioral_timeseries.create_timeseries(
            name = label_key['label'],
            description = label_name_map[label_key['label']],
            data = coords,
            timestamps = timestamps,
            unit = "pixels",
        )
    return behavioral_timeseries

def export_session_start(key):
    # estimate session start time from file metadata
    p_file = (exp.JoystickExperiment & key).get_path("p_params")
    # get sessions duration
    duration = (exp.JoystickExperiment.Trials & dict(**key, trial_id=99)).fetch1("t_end")
    file_stat = p_file.stat()
    session_start_time = file_stat.st_mtime - duration
    dt = datetime.datetime.fromtimestamp(session_start_time)
    # add local timezone info
    dt = dt.astimezone()
    return dt

def export_limb_features(key, side):
    # export extracted limb features
    features_to_export = ["velocity_x", "velocity_y", "velocity", "distance_x", "distance_y", "distance",
                          "bend_24", "rotation_24", "open_alt_24"]
    features_to_name = {
        "velocity_x": "velocity_x",
        "velocity_y": "velocity_y",
        "velocity": "velocity",
        "distance_x": "distance_x",
        "distance_y": "distance_y",
        "distance": "distance",
        "bend_24": "finger_bend",
        "rotation_24": "limb_rotation",
        "open_alt_24": "hand_aperture",
    }
    feature_units = {
        "velocity_x": "pixels/second",
        "velocity_y": "pixels/second",
        "velocity": "pixels/second",
        "distance_x": "pixels",
        "distance_y": "pixels",
        "distance": "pixels",
        "bend_24": "degrees",
        "rotation_24": "degrees",
        "open_alt_24": "degrees",
    }
    feature_descriptions = {
        "velocity_x": "Velocity of limb in x direction",
        "velocity_y": "Velocity of limb in y direction",
        "velocity": "L2 norm of limb velocity",
        "distance_x": "Distance of limb in x direction from joystick",
        "distance_y": "Distance of limb in y direction from joystick",
        "distance": "L2 norm of distance of limb from joystick",
        "bend_24": "Average bend of 2nd, 3rd and 4th finger",
        "rotation_24": "Overall rotation of the limb w.r.t. to vertical axis",
        "open_alt_24": "Aperture of the hand defined as angle between 2nd and 4th finger",
    }

    # load feature data
    key_limb = (pt.PawRecording.Hand & dict(**key, side=side)).fetch1('KEY')
    timestamps = (pt.Synchronisation.Hand & key_limb).fetch1("frame_timestamps")
    feature_matrix, names = (pt.Features.Hand & key_limb).fetch_feature_matrix(labels_to_include = features_to_export)

    # create BehavioralTimeSeries
    behavioral_timeseries = BehavioralTimeSeries(
        name = "TaskLimbFeatures" if side=='ipsi' else "SupportLimbFeatures",
    )
    for i, feature_name in enumerate(names):
        data = feature_matrix[:, i]
        # add data to BehavioralTimeSeries
        behavioral_timeseries.create_timeseries(
            name = features_to_name[feature_name],
            description = feature_descriptions[feature_name],
            data = data,
            timestamps = timestamps,
            unit = feature_units[feature_name],
        )
    return behavioral_timeseries

def export_trials(key, nwbfile):
    # create trials interface
    nwbfile.add_trial_column("successful", "Whether trial was successful")
    nwbfile.add_trial_column("autoreward", "Whether trial was autorewarded")
    nwbfile.add_trial_column("t_servo_in", "Time joystick moved into position")
    nwbfile.add_trial_column("t_cue", "Time cue was presented")
    nwbfile.add_trial_column("t_servo_out", "Time joystick moved out of position, corresponds to reward time in successful trials")

    trials = (exp.JoystickExperiment.Trials & key).fetch(as_dict=True)
    for trial in trials:
        nwbfile.add_trial(
            start_time = trial['t_start'],
            stop_time = trial['t_end'],
            successful = trial['successful'],
            autoreward = trial['autoreward'],
            t_servo_in = trial['t_servo_in'],
            t_cue = trial['t_cue'],
            t_servo_out = trial['t_servo_out'],
        )
    return nwbfile

def export_tasklimb_epochs(key):
    # create epochs for task limb segmentation
    key_ipsi = (pt.PawRecording.Hand & dict(**key, side='ipsi')).fetch1('KEY')
    epochs = (
        pt.MovementSegmentation.Epoch.proj("start_time", "end_time")
        * pt.EpochClassification.Epoch.proj("epoch_class")
        & key_ipsi
    ).fetch(as_dict=True, order_by="epoch_id")

    # create TimeIntervals
    tasklimb_epochs = TimeIntervals(
        name = "TaskLimbEpochs",
        description = "Segmented epochs of task limb movements, classified into rewarded, missed, other and excluded movements"
    )
    tasklimb_epochs.add_column("epoch_class", "Classification of the epoch into rewarded, missed, other and excluded movements")
    tasklimb_epochs.add_column("epoch_id", "Unique identifier of the epoch")
    for epoch in epochs:
        tasklimb_epochs.add_interval(
            start_time = epoch['start_time'],
            stop_time = epoch['end_time'],
            epoch_class = epoch['epoch_class'],
            epoch_id = epoch['epoch_id'],
        )
    return tasklimb_epochs

def export_supportlimb_epochs(key):
    # create epochs for support limb segmentation
    key_contra = (pt.PawRecording.Hand & dict(**key, side='contra')).fetch1('KEY')
    epochs = (
        pt.MovementSegmentation.Epoch.proj("start_time", "end_time")
        & key_contra
    ).fetch(as_dict=True, order_by="epoch_id")
    # create TimeIntervals
    supportlimb_epochs = TimeIntervals(
        name = "SupportLimbEpochs",
        description = "Segmented epochs of support limb movements"
    )
    supportlimb_epochs.add_column("epoch_id", "Unique identifier of the epoch")
    for epoch in epochs:
        supportlimb_epochs.add_interval(
            start_time = epoch['start_time'],
            stop_time = epoch['end_time'],
            epoch_id = epoch['epoch_id'],
        )
    return supportlimb_epochs

def export_widefield_imaging(key, nwbfile):
    # create device
    device = nwbfile.create_device(
        name='Microscope',
        description='Custom widefield imaging setup with tandem-lens macroscope and sCMOS camera',
        manufacturer='Brain Research Institute, University of Zurich',
    )
    # create optical channel
    optical_channel = OpticalChannel(
        name="GCaMP6f",
        description='Hemodynamics-corrected GCaMP6f fluorescence signal',
        emission_lambda=524.0,
    )

    # get n_frames, h, w
    n_frames, h, w, binning = (wf.WidefieldSession & key).fetch1("n_frames", "pixels_h", "pixels_w", "binning")

    # create imaging plane
    imaging_plane = nwbfile.create_imaging_plane(
        name='Dorsal Cortex',
        optical_channel=optical_channel,
        imaging_rate = 20.0,
        device=device,
        excitation_lambda=470.0,
        indicator='GCaMP6f',
        location='Dorsal cortex',
        description='Widefield calcium imaging of dorsal cortex through intact skull',
    )

    # create processing module for fluorescence data
    ophys_module = nwbfile.create_processing_module(
        name='ophys',
        description='Preprocessed and segmented widefield imaging data',
    )

    # get list of segmented ROIs
    rois = [
        "MOs-lateral", "MOs-medial", "MOp", "SSp-ll", "SSp-ul", "SSp-nosemouth", "SSp-bfd", "SSp-tr",
        "RSp-anterior", "RSP-posterior", "VISp", "VIS-medial", "VISa", "VISrl"
    ]
    rois_to_load = [r + "_contra" for r in rois] + [r + "_ipsi" for r in rois]
    roi_dffs, roi_ids = (
        wf.AllenSegmentation2 & dict(**key, wf_param_id=6)
    ).load_rois(roi_list=rois_to_load)
    roi_dff_matrix = np.stack(roi_dffs).T.astype(np.float32)  # shape (n_frames, n_rois)

    roi_ids_new = []
    for roi_id in roi_ids:
        if roi_id.endswith("_contra"):
            roi_ids_new.append(roi_id.replace("_contra", "_ipsilesional"))
        elif roi_id.endswith("_ipsi"):
            roi_ids_new.append(roi_id.replace("_ipsi", "_contralesional"))
        else:
            roi_ids_new.append(roi_id)

    timestamps = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
    # add each ROI as a normal timeseries
    for i, roi_id in enumerate(roi_ids_new):
        roi_dff = roi_dff_matrix[:, i]
        timeseries = TimeSeries(
            name = f"{roi_id}",
            data = roi_dff,
            unit = "dF/F",
            timestamps = timestamps,
            description = f"Hemodynamics-corrected dFF signal of ROI {roi_id}",
        )
        ophys_module.add_data_interface(timeseries)

    return nwbfile


if __name__ == "__main__":
    print("Starting NWB export...")
    # generate list of keys for export
    keys = (
        exp.DaysFromStrokeNorm
        * exp.StrokeGroup.proj(stroke_group='group')
        * exp.ExperimentalPhase
        & [f"days_from_stroke_norm = {d}" for d in [-3, -2, -1, 3, 7, 14, 21, 28]]
        & "phase != 'Learning'"
        & "stroke_group != 'Learning'"
        & "mouse_id > 40"
    ).fetch('KEY')
    print(f"Number of sessions to export: {len(keys)}")

    # create output directory
    output_dir = Path("~/neurophys_3/paper_submission_data/").expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)


    for key in tqdm(keys, desc="Exporting NWB files"):
        # create NWB file
        nwbfile = NWBFile(
            # Required fields
            session_description = export_session_description(key),
            identifier = str(uuid4()),
            session_start_time = export_session_start(key),
            # Human-readable ID
            experimenter = "Panzeri, Matteo",
            lab = "Junior Group Wahl, Helmchen Lab",
            institution = "Brain Research Institute, University of Zurich, Switzerland",
            experiment_description = "Chronic widefield calcium imaging of dorsal cortex during a reach-to-grasp joystick task, before and after large ischemic strokes. Behavioral parameters recorded are keypoint tracking of both forelimbs.",
            keywords = ["widefield calcium imaging", "motor behavior", "stroke", "ischemia", "cortex", "chronic recording"],
            related_publications = None,
        )

        # add subject info
        nwbfile.subject = export_subject(key)

        # create behavioral processing module
        behavior_module = nwbfile.create_processing_module(
            name = "behavior",
            description = "Processed behavioral data",
        )

        # add deeplabcut tracking data for both limbs
        dlc_timeseries_ipsi = export_limb_DLC(key, side='ipsi')
        dlc_timeseries_contra = export_limb_DLC(key, side='contra')
        features_ipsi = export_limb_features(key, side='ipsi')
        features_contra = export_limb_features(key, side='contra')
        # add to behavior module
        behavior_module.add_data_interface(dlc_timeseries_ipsi)
        behavior_module.add_data_interface(dlc_timeseries_contra)
        behavior_module.add_data_interface(features_ipsi)
        behavior_module.add_data_interface(features_contra)

        # add trials
        nwbfile = export_trials(key, nwbfile)

        # add task limb epochs
        tasklimb_epochs = export_tasklimb_epochs(key)
        behavior_module.add_data_interface(tasklimb_epochs)
        # add support limb epochs
        supportlimb_epochs = export_supportlimb_epochs(key)
        behavior_module.add_data_interface(supportlimb_epochs)

        # add segmented widefield imaging data
        nwbfile = export_widefield_imaging(key, nwbfile)

        # create output file path
        filename = f"M{key['mouse_id']:03d}_D{key['day'].strftime('%Y%m%d')}_nwb_export.nwb"
        output_path = output_dir / filename
        # write NWB file
        with NWBHDF5IO(output_path, 'w') as io:
            io.write(nwbfile)
    print("NWB export completed.")
