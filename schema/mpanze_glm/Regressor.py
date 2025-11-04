"""
Defines the Regressor table for the mpanze_glm schema.
This table defines the types of regressors that models can use for fitting.
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# dependencies
from schema.mpanze_widefield_refactor import mpanze_widefield_refactor as wf
from schema.mpanze_paw_tracking_refactor import mpanze_paw_tracking_refactor as pt
from schema import mpanze_face_tracking as ft
from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp
import numpy as np
from scipy.ndimage import gaussian_filter1d

# instantiate the schema
schema = dj.schema('mpanze_glm', locals(), create_tables=True)

@schema
class Regressor(dj.Lookup):
    definition = """ # defines regressors used for GLM
    reg_name            : varchar(64)               # name of the regressor
    ---
    reg_description     : varchar(255)              # description of the regressor
    reg_type            : enum('event', 'analog')   # type of regressor
    reg_dim             : int                       # dimension of the regressor
    reg_schema          : varchar(255)              # schema of the source table
    reg_source          : varchar(64)               # name of the source table
    reg_field           : varchar(64)               # name of the source field
    """
    contents = [
        ('TaskLimb0', 'Event regressor representing task (ipsi) limb onsets and offsets', 'event', 2, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time, end_time'),
        ('TaskLimbOnsets', 'Event regressor representing task (ipsi) limb onsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time'),
        ('TaskLimbOffsets', 'Event regressor representing task (ipsi) limb offsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'end_time'),
        ('SupportLimb0', 'Event regressor representing support (contra) limb onsets and offsets', 'event', 2, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time, end_time'),
        ('SupportLimbOnsets', 'Event regressor representing support (contra) limb onsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time'),
        ('SupportLimbOffsets', 'Event regressor representing support (contra) limb offsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'end_time'),
        ('RewardedTaskLimb0', 'Event regressor representing rewarded task (ipsi) limb onsets and offsets', 'event', 2, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time, end_time'),
        ('RewardedTaskLimbOnsets', 'Event regressor representing rewarded task (ipsi) limb onsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time'),
        ('RewardedTaskLimbOffsets', 'Event regressor representing rewarded task (ipsi) limb offsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'end_time'),
        ('MissTaskLimb0', 'Event regressor representing missed task (ipsi) limb onsets and offsets', 'event', 2, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time, end_time'),
        ('MissTaskLimbOnsets', 'Event regressor representing missed task (ipsi) limb onsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time'),
        ('MissTaskLimbOffsets', 'Event regressor representing missed task (ipsi) limb offsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'end_time'),
        ('OtherTaskLimb0', 'Event regressor representing other task (ipsi) limb onsets and offsets', 'event', 2, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time, end_time'),
        ('OtherTaskLimbOnsets', 'Event regressor representing other task (ipsi) limb onsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time'),
        ('OtherTaskLimbOffsets', 'Event regressor representing other task (ipsi) limb offsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'end_time'),
        ('ExcludedTaskLimb0', 'Event regressor representing excluded task (ipsi) limb onsets and offsets', 'event', 2, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time, end_time'),
        ('ExcludedTaskLimbOnsets', 'Event regressor representing excluded task (ipsi) limb onsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'start_time'),
        ('ExcludedTaskLimbOffsets', 'Event regressor representing excluded task (ipsi) limb offsets only', 'event', 1, 'mpanze_paw_tracking_refactor', 'MovementSegmentation.Epoch', 'end_time'),
        ('ValveOpeningEvents', 'Valve opening events (reward + autorewards)', 'event', 1, 'mpanze_exp', 'JoystickExperiment.Trials', 't_servo_out'),
        ('CueEvents', 'Auditory cue onsets, indicating the start of the trial', 'event', 1, 'mpanze_exp', 'JoystickExperiment.Trials', 't_cue'),
        ('NoseNME', 'Normalised Motion Energy from nose ROI', 'analog', 1, 'mpanze_face_tracking', 'NormalizedMotionEnergy', 'nme_nose'),
        ('MouthNME', 'Normalised Motion Energy from mouth ROI', 'analog', 1, 'mpanze_face_tracking', 'NormalizedMotionEnergy', 'nme_mouth'),
        ('WhiskNME', 'Normalised Motion Energy from whisk ROI', 'analog', 1, 'mpanze_face_tracking', 'NormalizedMotionEnergy', 'nme_whisk'),
        ('PupilRadius', 'Pupil radius extracted from face camera', 'analog', 1, 'mpanze_face_tracking', 'PupilFit', 'radius'),
        ('FaceSVDRaw', '(top 100) SVD components from raw face video', 'analog', 100, 'mpanze_face_tracking', 'FaceSVD', 'svt_raw'),
        ('FaceSVDME', '(top 100) SVD components from motion energy face video', 'analog', 100, 'mpanze_face_tracking', 'FaceSVD', 'svt_me'),
        ('HandFeaturesIpsi', 'Hand features (ipsi)', 'analog', 10, 'mpanze_paw_tracking_refactor', 'Features', 'features'),
        ('HandFeaturesContra', 'Hand features (contra)', 'analog', 10, 'mpanze_paw_tracking_refactor', 'Features', 'features'),
        ('HandCoordsIpsi', 'Hand coordinates (ipsi)', 'analog', 10, 'mpanze_paw_tracking_refactor', 'FilteredDLC', 'label'),
        ('HandCoordsContra', 'Hand coordinates (contra)', 'analog', 10, 'mpanze_paw_tracking_refactor', 'FilteredDLC', 'label'),
        ('MotorFeaturesIpsi', 'Motor features (ipsi)', 'analog', 7, 'mpanze_paw_tracking_refactor', 'Features', 'features'),
        ('MotorFeaturesContra', 'Motor features (contra)', 'analog', 7, 'mpanze_paw_tracking_refactor', 'Features', 'features')
    ]

    def load_1(self, key):
        assert len(self)==1, "select only one regressor at a time"
        reg_name = self.fetch1('reg_name')
        # big ugly if statement to load the correct regressor
        if reg_name == 'TaskLimb0':
            return self.load_TaskLimb0(key)
        elif reg_name == 'TaskLimbOnsets':
            return self.load_TaskLimbOnsets(key)
        elif reg_name == 'TaskLimbOffsets':
            return self.load_TaskLimbOffsets(key)
        elif reg_name == 'SupportLimb0':
            return self.load_SupportLimb0(key)
        elif reg_name == 'SupportLimbOnsets':
            return self.load_SupportLimbOnsets(key)
        elif reg_name == 'SupportLimbOffsets':
            return self.load_SupportLimbOffsets(key)
        elif reg_name == 'RewardedTaskLimb0':
            return self.load_RewardedTaskLimb0(key)
        elif reg_name == 'RewardedTaskLimbOnsets':
            return self.load_RewardedTaskLimbOnsets(key)
        elif reg_name == 'RewardedTaskLimbOffsets':
            return self.load_RewardedTaskLimbOffsets(key)
        elif reg_name == 'MissTaskLimb0':
            return self.load_MissTaskLimb0(key)
        elif reg_name == 'MissTaskLimbOnsets':
            return self.load_MissTaskLimbOnsets(key)
        elif reg_name == 'MissTaskLimbOffsets':
            return self.load_MissTaskLimbOffsets(key)
        elif reg_name == 'OtherTaskLimb0':
            return self.load_OtherTaskLimb0(key)
        elif reg_name == 'OtherTaskLimbOnsets':
            return self.load_OtherTaskLimbOnsets(key)
        elif reg_name == 'OtherTaskLimbOffsets':
            return self.load_OtherTaskLimbOffsets(key)
        elif reg_name == 'ExcludedTaskLimb0':
            return self.load_ExcludedTaskLimb0(key)
        elif reg_name == 'ExcludedTaskLimbOnsets':
            return self.load_ExcludedTaskLimbOnsets(key)
        elif reg_name == 'ExcludedTaskLimbOffsets':
            return self.load_ExcludedTaskLimbOffsets(key)
        elif reg_name == 'ValveOpeningEvents':
            return self.load_ValveOpeningEvents(key)
        elif reg_name == 'CueEvents':
            return self.load_CueEvents(key)
        elif reg_name == 'NoseNME':
            return self.load_NoseNME(key)
        elif reg_name == 'MouthNME':
            return self.load_MouthNME(key)
        elif reg_name == 'WhiskNME':
            return self.load_WhiskNME(key)
        elif reg_name == 'PupilRadius':
            return self.load_PupilRadius(key)
        elif reg_name == 'FaceSVDRaw':
            return self.load_FaceSVDRaw(key)
        elif reg_name == 'FaceSVDME':
            return self.load_FaceSVDME(key)
        elif reg_name == 'HandFeaturesIpsi':
            return self.load_HandFeaturesIpsi(key)
        elif reg_name == 'HandFeaturesContra':
            return self.load_HandFeaturesContra(key)
        elif reg_name == 'HandCoordsIpsi':
            return self.load_HandCoordsIpsi(key)
        elif reg_name == 'HandCoordsContra':
            return self.load_HandCoordsContra(key)
        elif reg_name == 'MotorFeaturesIpsi':
            return self.load_MotorFeaturesIpsi(key)
        elif reg_name == 'MotorFeaturesContra':
            return self.load_MotorFeaturesContra(key)
        else:
            raise NotImplementedError(f"Regressor {reg_name} not implemented")

    @staticmethod
    def load_TaskLimb0(key):
        # check if key there is an entry for task limb segmentation
        key_tasklimb = (pt.MovementSegmentation.Hand & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts, ends
        starts, ends = (pt.MovementSegmentation.Epoch & key_tasklimb).fetch('start_time', 'end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 2), dtype=np.float32)
        regressor[frames_start, 0] = 1  # onset
        regressor[frames_end, 1] = 1  # offset
        # return the regressor
        return regressor
    
    @staticmethod
    def load_TaskLimbOnsets(key):
        # check if key there is an entry for task limb segmentation
        key_tasklimb = (pt.MovementSegmentation.Hand & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts
        starts = (pt.MovementSegmentation.Epoch & key_tasklimb).fetch('start_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_start] = 1  # onset
        # return the regressor
        return regressor
    
    @staticmethod
    def load_TaskLimbOffsets(key):
        # check if key there is an entry for task limb segmentation
        key_tasklimb = (pt.MovementSegmentation.Hand & dict(**key, side='ipsi')).fetch1('KEY')
        # load ends
        ends = (pt.MovementSegmentation.Epoch & key_tasklimb).fetch('end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_end] = 1  # offset
        # return the regressor
        return regressor

    @staticmethod
    def load_SupportLimb0(key):
        # check if key there is an entry for support limb segmentation
        key_supportlimb = (pt.MovementSegmentation.Hand & dict(**key, side='contra')).fetch1('KEY')
        # load starts, ends
        starts, ends = (pt.MovementSegmentation.Epoch & key_supportlimb).fetch('start_time', 'end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 2), dtype=np.float32)
        regressor[frames_start, 0] = 1  # onset
        regressor[frames_end, 1] = 1  # offset
        # return the regressor
        return regressor
    
    @staticmethod
    def load_SupportLimbOnsets(key):
        # check if key there is an entry for support limb segmentation
        key_supportlimb = (pt.MovementSegmentation.Hand & dict(**key, side='contra')).fetch1('KEY')
        # load starts
        starts = (pt.MovementSegmentation.Epoch & key_supportlimb).fetch('start_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_start] = 1  # onset
        # return the regressor
        return regressor
    
    @staticmethod
    def load_SupportLimbOffsets(key):
        # check if key there is an entry for support limb segmentation
        key_supportlimb = (pt.MovementSegmentation.Hand & dict(**key, side='contra')).fetch1('KEY')
        # load ends
        ends = (pt.MovementSegmentation.Epoch & key_supportlimb).fetch('end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_end] = 1  # offset
        # return the regressor
        return regressor

    @staticmethod
    def load_RewardedTaskLimb0(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts, ends
        starts, ends = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='rewarded')).fetch('start_time', 'end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 2), dtype=np.float32)
        regressor[frames_start, 0] = 1  # onset
        regressor[frames_end, 1] = 1  # offset
        # return the regressor
        return regressor
    
    @staticmethod
    def load_RewardedTaskLimbOnsets(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts
        starts = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='rewarded')).fetch('start_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_start] = 1
        # return the regressor
        return regressor
    
    @staticmethod
    def load_RewardedTaskLimbOffsets(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load ends
        ends = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='rewarded')).fetch('end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_end] = 1
        # return the regressor
        return regressor

    @staticmethod
    def load_MissTaskLimb0(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts, ends
        starts, ends = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='miss')).fetch('start_time', 'end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 2), dtype=np.float32)
        regressor[frames_start, 0] = 1  # onset
        regressor[frames_end, 1] = 1  # offset
        # return the regressor
        return regressor
    
    @staticmethod
    def load_MissTaskLimbOnsets(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts
        starts = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='miss')).fetch('start_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_start] = 1
        # return the regressor
        return regressor
        
    @staticmethod
    def load_MissTaskLimbOffsets(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load ends
        ends = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='miss')).fetch('end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_end] = 1
        # return the regressor
        return regressor

    @staticmethod
    def load_OtherTaskLimb0(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts, ends
        starts, ends = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='other')).fetch('start_time', 'end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 2), dtype=np.float32)
        regressor[frames_start, 0] = 1  # onset
        regressor[frames_end, 1] = 1  # offset
        # return the regressor
        return regressor
    
    @staticmethod
    def load_OtherTaskLimbOnsets(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts
        starts = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='other')).fetch('start_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_start] = 1
        # return the regressor
        return regressor
    
    @staticmethod
    def load_OtherTaskLimbOffsets(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load ends
        ends = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='other')).fetch('end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_end] = 1
        # return the regressor
        return regressor

    @staticmethod
    def load_ExcludedTaskLimb0(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts, ends
        starts, ends = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='excluded')).fetch('start_time', 'end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 2), dtype=np.float32)
        regressor[frames_start, 0] = 1  # onset
        regressor[frames_end, 1] = 1  # offset
        # return the regressor
        return regressor
    
    @staticmethod
    def load_ExcludedTaskLimbOnsets(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load starts
        starts = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='excluded')).fetch('start_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_start = np.searchsorted(frames, starts)
        n_frames = len(frames)
        frames_start = np.clip(frames_start, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_start] = 1
        # return the regressor
        return regressor
    
    @staticmethod
    def load_ExcludedTaskLimbOffsets(key):
        # check if there are available entries
        key_tasklimb = (pt.MovementSegmentation.Hand * pt.EpochClassification & dict(**key, side='ipsi')).fetch1('KEY')
        # load ends
        ends = (pt.MovementSegmentation.Epoch * pt.EpochClassification.Epoch & dict(**key_tasklimb, epoch_class='excluded')).fetch('end_time')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_end = np.searchsorted(frames, ends)
        n_frames = len(frames)
        frames_end = np.clip(frames_end, 0, n_frames - 1)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_end] = 1
        # return the regressor
        return regressor

    @staticmethod
    def load_ValveOpeningEvents(key):
        # get the valve opening times
        valveopening = (exp.JoystickExperiment.Trials & key & "successful=1 OR autoreward=1").fetch('t_servo_out')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_valveopening = np.searchsorted(frames, valveopening)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_valveopening, 0] = 1
        # return the regressor
        return regressor
    
    @staticmethod
    def load_CueEvents(key):
        # get the cue onset times
        cues = (exp.JoystickExperiment.Trials & key).fetch('t_cue')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        frames_cue = np.searchsorted(frames, cues)
        # create the regressor
        regressor = np.zeros((len(frames), 1), dtype=np.float32)
        regressor[frames_cue, 0] = 1
        # return the regressor
        return regressor
    
    @staticmethod
    def load_NoseNME(key):
        # load normalized motion energy for nose
        frames_nme = (ft.Synchronisation & key).fetch1("frame_timestamps")
        nme_nose = (ft.NormalizedMotionEnergy & key).fetch1('nme_nose')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        # interpolate the NME to the frame timestamps
        nme_nose_interp = np.interp(frames, frames_nme, nme_nose).astype(np.float32).reshape(-1, 1)
        return nme_nose_interp

    @staticmethod
    def load_MouthNME(key):
        # load normalized motion energy for mouth
        frames_nme = (ft.Synchronisation & key).fetch1("frame_timestamps")
        nme_mouth = (ft.NormalizedMotionEnergy & key).fetch1('nme_mouth')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        # interpolate the NME to the frame timestamps
        nme_mouth_interp = np.interp(frames, frames_nme, nme_mouth).astype(np.float32).reshape(-1, 1)
        return nme_mouth_interp
    
    @staticmethod
    def load_WhiskNME(key):
        # load normalized motion energy for whisking
        frames_nme = (ft.Synchronisation & key).fetch1("frame_timestamps")
        nme_whisk = (ft.NormalizedMotionEnergy & key).fetch1('nme_whisk')
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        # interpolate the NME to the frame timestamps
        nme_whisk_interp = np.interp(frames, frames_nme, nme_whisk).astype(np.float32).reshape(-1, 1)
        return nme_whisk_interp

    @staticmethod
    def load_PupilRadius(key):
        frames_pupil = (ft.Synchronisation & key).fetch1("frame_timestamps")
        radius = (ft.PupilFit & key).fetch1('radius')
        radius = gaussian_filter1d(radius, sigma=5)
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        # interpolate the pupil radius to the frame timestamps
        radius_interp = np.interp(frames, frames_pupil, radius).astype(np.float32).reshape(-1, 1)
        # z-score the variable using median absolute deviation
        med = np.median(radius_interp)
        mad = np.median(np.abs(radius_interp - med))
        regressor_z = (radius_interp - med) / (mad * 1.4826)
        regressor_z = np.clip(regressor_z, -7.5, 7.5)  # clip to avoid extreme values
        return regressor_z

    @staticmethod
    def load_FaceSVDRaw(key):
        # load SVD components from raw face video
        frames_svd = (ft.Synchronisation & key).fetch1("frame_timestamps")
        svt_raw = (ft.FaceSVD & key).fetch1('svt_raw')[:100].T  # take top 100 components
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        # interpolate the SVD components to the frame timestamps
        regressor_interp = np.zeros((len(frames), 100), dtype=np.float32)
        for i in range(100):
            regressor_interp[:, i] = np.interp(frames, frames_svd, svt_raw[:,i])
        # z-score the variable
        mean = np.mean(regressor_interp, axis=0)
        std = np.std(regressor_interp, axis=0)
        regressor_z = (regressor_interp - mean) / std
        return regressor_z
    
    @staticmethod
    def load_FaceSVDME(key):
        # load SVD components from motion energy face video
        frames_svd = (ft.Synchronisation & key).fetch1("frame_timestamps")
        svt_me = (ft.FaceSVD & key).fetch1('svt_me')[:100].T
        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")
        # interpolate the SVD components to the frame timestamps
        regressor_interp = np.zeros((len(frames), 100), dtype=np.float32)
        for i in range(100):
            regressor_interp[:, i] = np.interp(frames, frames_svd, svt_me[:,i])
        # z-score the variable
        mean = np.mean(regressor_interp, axis=0)
        std = np.std(regressor_interp, axis=0)
        regressor_z = (regressor_interp - mean) / std
        return regressor_z

    @staticmethod
    def load_HandFeaturesIpsi(key):
        # load feature matrix
        hand_key = (pt.PawRecording.Hand & dict(**key, side='ipsi')).fetch1('KEY')
        features = (pt.Features.Hand & hand_key).fetch_feature_matrix()
        frames_features = (pt.Synchronisation.Hand & hand_key).fetch1("frame_timestamps")

        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")

        # compress the feature matrix to 10 dims
        from sklearn.decomposition import PCA
        features = PCA(n_components=10).fit_transform(features)

        # interpolate the features to the frame timestamps
        regressor_interp = np.zeros((len(frames), 10), dtype=np.float32)
        for i in range(10):
            regressor_interp[:, i] = np.interp(frames, frames_features, features[:, i])
        
        return regressor_interp
    
    @staticmethod
    def load_HandFeaturesContra(key):
        # load feature matrix
        hand_key = (pt.PawRecording.Hand & dict(**key, side='contra')).fetch1('KEY')
        features = (pt.Features.Hand & hand_key).fetch_feature_matrix()
        frames_features = (pt.Synchronisation.Hand & hand_key).fetch1("frame_timestamps")

        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")

        # compress the feature matrix to 10 dims
        from sklearn.decomposition import PCA
        features = PCA(n_components=10).fit_transform(features)

        # interpolate the features to the frame timestamps
        regressor_interp = np.zeros((len(frames), 10), dtype=np.float32)
        for i in range(10):
            regressor_interp[:, i] = np.interp(frames, frames_features, features[:, i])
        
        return regressor_interp
    
    @staticmethod
    def load_HandCoordsIpsi(key):
        key = (pt.PawRecording.Hand & dict(**key, side='ipsi')).fetch1('KEY')
        coords = (pt.FilteredDLC.Hand & key).fetch_coordinate_matrix()
        frames_coords = (pt.Synchronisation.Hand & key).fetch1("frame_timestamps")

        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")

        # compress the coordinate matrix to 10 dims
        from sklearn.decomposition import PCA
        coords = PCA(n_components=10).fit_transform(coords)
        # interpolate the coordinates to the frame timestamps
        regressor_interp = np.zeros((len(frames), 10), dtype=np.float32)
        for i in range(10):
            regressor_interp[:, i] = np.interp(frames, frames_coords, coords[:, i])
        return regressor_interp

    @staticmethod
    def load_HandCoordsContra(key):
        key = (pt.PawRecording.Hand & dict(**key, side='contra')).fetch1('KEY')
        coords = (pt.FilteredDLC.Hand & key).fetch_coordinate_matrix()
        frames_coords = (pt.Synchronisation.Hand & key).fetch1("frame_timestamps")

        # fetch wf synchronisation
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")

        # compress the coordinate matrix to 10 dims
        from sklearn.decomposition import PCA
        coords = PCA(n_components=10).fit_transform(coords)
        # interpolate the coordinates to the frame timestamps
        regressor_interp = np.zeros((len(frames), 10), dtype=np.float32)
        for i in range(10):
            regressor_interp[:, i] = np.interp(frames, frames_coords, coords[:, i])
        return regressor_interp

    @staticmethod
    def load_MotorFeaturesIpsi(key):
        key = (pt.PawRecording.Hand & dict(**key, side='ipsi')).fetch1('KEY')
        labels_to_include = ["velocity_x", "velocity_y", "velocity", "acceleration",
                             "bend_24", "open_alt_24", "rotation_24"]
        features, _ = (
            pt.Features.Hand & key
        ).fetch_feature_matrix(labels_to_include=labels_to_include)

        frames_features = (pt.Synchronisation.Hand & key).fetch1("frame_timestamps")
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")

        # interpolate
        regressor_interp = np.zeros((len(frames), 7), dtype=np.float32)
        for i in range(7):
            regressor_interp[:, i] = np.interp(frames, frames_features, features[:, i])
        return regressor_interp
    
    @staticmethod
    def load_MotorFeaturesContra(key):
        key = (pt.PawRecording.Hand & dict(**key, side='contra')).fetch1('KEY')
        labels_to_include = ["velocity_x", "velocity_y", "velocity", "acceleration",
                             "bend_24", "open_alt_24", "rotation_24"]
        features, _ = (
            pt.Features.Hand & key
        ).fetch_feature_matrix(labels_to_include=labels_to_include)

        frames_features = (pt.Synchronisation.Hand & key).fetch1("frame_timestamps")
        frames = (wf.Synchronisation & key).fetch1("frame_timestamps_blue")

        # interpolate
        regressor_interp = np.zeros((len(frames), 7), dtype=np.float32)
        for i in range(7):
            regressor_interp[:, i] = np.interp(frames, frames_features, features[:, i])
        return regressor_interp
