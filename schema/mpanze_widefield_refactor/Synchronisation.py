"""
Synchronise data to global session time
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema.mpanze_widefield_refactor.WidefieldSession import WidefieldSession
from schema.mpanze_exp_refactor.JoystickReadouts import JoystickReadouts

# other dependencies
import numpy as np
import datetime

# instantiate the schema
schema = dj.schema('mpanze_widefield_refactor', locals(), create_tables=True)

@schema
class Synchronisation(dj.Computed):
    definition = """ # provides frame timestamps for each file in session
    -> WidefieldSession
    -> JoystickReadouts
    ---
    frame_timestamps_blue   : longblob          # 1-d array of timestamps in seconds of each frame, blue channel
    frame_timestamps_uv     : longblob          # 1-d array of timestamps in seconds of each frame, uv channel
    fps                     : float             # estimated framerate per channel, in Hz
    dt_median               : float             # median frame duration in seconds
    n_truncated_frames      : int               # number of truncated frames
    """

    _key_source = WidefieldSession * JoystickReadouts

    def make(self, key):
        # get number of frames from recording file
        n_frames = (WidefieldSession & key).fetch1('n_frames')
        frame_timestamps, dt_median = (JoystickReadouts.Sync & key).fetch1('frame_timestamps', 'dt_median')

        # funky, hacky solution for a subset of recordings where the sync signal is from 40Hz cable and not 20Hz cable
        if (len(frame_timestamps) // 2 > n_frames - 6) and (len(frame_timestamps) // 2 < n_frames + 6):
            print("WARNING: sync signal is from 40Hz cable, not 20Hz cable")
            print("KEY: ", key)
            # take every second frame
            frame_timestamps = frame_timestamps[::2]
            dt_median = dt_median * 2

        # special case: -2 frame detection, in mouse 72, '2024-10-21', truncate timestamps
        if (key["mouse_id"] == 72) and (key["day"] == datetime.date(2024, 10, 21)):
            print("WARNING: special case for mouse 72, '2024-10-21'")
            print("KEY: ", key)
            frame_timestamps = frame_timestamps[:n_frames]
            n_frames_diff = 0
        
        # sync signal is truncated, so we need to add frames at the end, truncation should miss at most 2 frames
        n_frames_diff = n_frames - len(frame_timestamps)
    
        assert n_frames_diff >= 0, f"too many frames detected {n_frames_diff}"             # more frames detected than in recording, should not happen
        assert n_frames_diff < 4, f"too few frames detected from sync {n_frames_diff}"    # too few frames detected, should not happen
        # add frames at the end
        frame_timestamps = np.append(frame_timestamps, frame_timestamps[-1] + np.arange(1, n_frames_diff+1) * dt_median)
        assert len(frame_timestamps) == n_frames, "incorrect number of frames detected"     # final sanity check

        # get timestamps of uv channel
        frame_timestamps_uv = frame_timestamps + dt_median/2

        # insert entry
        entry = dict(
            **key, 
            frame_timestamps_blue=frame_timestamps, 
            frame_timestamps_uv=frame_timestamps_uv,
            fps=1/dt_median,
            dt_median=dt_median, 
            n_truncated_frames=n_frames_diff,
        )
        self.insert1(entry)
