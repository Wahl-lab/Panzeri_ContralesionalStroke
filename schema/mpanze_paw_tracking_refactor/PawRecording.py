"""
Information about the behavioral recordings of the paws
"""

# connect to the database
import login
login.connect()

# imports
import datajoint as dj
from util import pathfinding
import numpy as np

# optional imports
from util.optional_import import import_optional
cv2 = import_optional('cv2', behavior='warn')

# import table dependencies
from schema import common_exp, common_mice
from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp

# instantiate the schema
schema = dj.schema('mpanze_paw_tracking_refactor', locals(), create_tables=True)

@schema
class PawRecording(dj.Imported):
    definition = """ # Basic info about the paw tracking session
    -> common_exp.Session
    ---
    -> exp.Handedness
    """
    # include only joytick experiments from batch 7 onwards, as these are formatted as single files
    _key_source = common_exp.Session * common_mice.Mouse.proj(batch='batch') & dict(username='mpanze', task="Joystick_with_cue") & "batch>=7"

    class Hand(dj.Part):
        definition = """ # Recording info for each hand
        -> PawRecording
        hand                : enum('L', 'R')            # left or right forelimb of the mouse
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        n_frames            : int                       # number of frames in the recording
        pixels_w            : int                       # width of the recording in pixels (2nd matrix dimension)
        pixels_h            : int                       # height of the recording in pixels (1st matrix dimension)
        fps                 : float                     # frame rate of the recording in Hz
        file_path           : varchar(512)              # relative path to the raw data file
        """
        def get_path(self, as_posix=False):
            assert len(self) == 1, "only one entry allowed"
            p = pathfinding.get_absolute_paths(self, "file_path")
            assert len(p) == 1, f"Multiple paths found for {self.fetch1('KEY')}"
            return p[0].as_posix() if as_posix else p[0]

        def get_paths(self, as_posix=False):
            for key in self.fetch("KEY"):
                yield (self & key).get_path(as_posix=as_posix)
    
    def make(self, key):
        """
        populate PawRecording table for batch 7 and later
        """
        # find raw behavioral files - should be only files in folder
        p_right = (common_exp.Session & key).glob("*_master_resize.avi")
        assert len(p_right) == 1, f"Multiple raw right paw files found for session {key}"
        p_right = p_right[0]
        p_left = (common_exp.Session & key).glob("*_slave_flip_resize.avi")
        assert len(p_left) == 1, f"Multiple raw left paw files found for session {key}"
        p_left = p_left[0]

        handedness = (exp.Handedness & key).fetch1("handedness")

        # create right paw entry
        entry_right = dict(
            **key,
            hand='R',
            side='ipsi' if handedness == 'R' else 'contra'
        )
        cap = cv2.VideoCapture(p_right.as_posix())
        assert cap.isOpened(), f"Error opening video file {p_right}"
        entry_right["fps"] = cap.get(cv2.CAP_PROP_FPS)
        entry_right["n_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        entry_right["pixels_w"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        entry_right["pixels_h"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        entry_right["file_path"] = (common_exp.Session & key).get_relative_path(p_right).as_posix()
        cap.release()

        # create left paw entry
        entry_left = dict(
            **key,
            hand='L',
            side='ipsi' if handedness == 'L' else 'contra'
        )
        cap = cv2.VideoCapture(p_left.as_posix())
        assert cap.isOpened(), f"Error opening video file {p_left}"
        entry_left["fps"] = cap.get(cv2.CAP_PROP_FPS)
        entry_left["n_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        entry_left["pixels_w"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        entry_left["pixels_h"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        entry_left["file_path"] = (common_exp.Session & key).get_relative_path(p_left).as_posix()
        cap.release()

        # insert data
        self.insert1(key)
        self.Hand.insert([entry_right, entry_left])

@schema
class Synchronisation(dj.Computed):
    definition = """ # provides frame timestamps for each behavioral recording
    -> PawRecording
    -> exp.JoystickReadouts
    ---
    """
    class Hand(dj.Part):
        definition = """ # synchronisation info for each hand
        -> Synchronisation
        hand                : enum('L', 'R')            # left or right forelimb of the mouse
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        frame_timestamps    : longblob                  # timestamps of each frame in the behavioral recording
        """

    def make(self, key):
        # fetch global synchronisation signal
        t_sync = (exp.JoystickReadouts.Data & key).fetch1("t")

        # compute timestamps for each hand
        hand_entries = []
        for hand_key in (PawRecording.Hand & key).fetch("KEY"):
            side, n_frames = (PawRecording.Hand & hand_key).fetch1("side", "n_frames")
            hand_entry = dict(**hand_key, side=side)
            frame_timestamps = np.linspace(t_sync[0], t_sync[-1], n_frames)
            hand_entry["frame_timestamps"] = frame_timestamps
            hand_entries.append(hand_entry)
        
        # insert data
        self.insert1(key)
        self.Hand.insert(hand_entries)

@schema
class JoystickPosition(dj.Manual):
    definition = """ # location of joystick in frame
    -> PawRecording
    ---
    """
    class Hand(dj.Part):
        definition = """ # location of joystick in frame for each hand
        -> JoystickPosition
        hand                : enum('L', 'R')            # left or right forelimb of the mouse
        ---
        side                : enum('ipsi', 'contra')    # ipsilateral or contralateral to the task hand
        x                   : float                     # x position of joystick tip in pixels
        y                   : float                     # y position of joystick tip in pixels
        """
