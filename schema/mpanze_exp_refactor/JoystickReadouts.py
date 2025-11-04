"""
Sensor readouts from joystick and touch sensor
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema.mpanze_exp_refactor.JoystickExperiment import JoystickExperiment

# other dependencies
import numpy as np
from skimage import measure

# instantiate the schema
schema = dj.schema('mpanze_exp_refactor', locals(), create_tables=True)

@schema
class JoystickReadouts(dj.Computed):
    definition = """ # table for storing sensor readouts from joystick
    -> JoystickExperiment
    ---
    """
    _key_source = JoystickExperiment

    class Data(dj.Part):
        definition = """ # raw readouts from daq file
        -> JoystickReadouts
        ---
        t           : longblob   # timestamps as seconds relative to session start
        x           : longblob   # x position of joystick
        y           : longblob   # y position of joystick
        touch       : longblob   # touch sensor readout, binarized
        wf_sync     : longblob   # synchronization signal from widefield camera, binarized
        """

    class Sync(dj.Part):
        definition = """ # conversion of sync signal to timestamps
        -> JoystickReadouts
        ---
        frame_timestamps    : longblob      # timestamps of each frame in the widefield camera
        exp_median          : float         # median exposure time for each widefield frame
        exp_std             : float         # standard deviation of frame exposure times
        dt_median           : float         # median time between frames
        dt_std              : float         # standard deviation of time between frames
        n_skipped           : int           # number of skipped frames
        skipped_timestamps  : longblob      # timestamps of skipped frames
        """

    def make(self, key):
        # find *daq.csv file
        p_file = (JoystickExperiment & key).get_path("p_daq")
        
        # extract raw data
        data = np.loadtxt(p_file, delimiter=',')
        t = data[:, 0]
        x = data[:, 1]
        y = data[:, 2]
        touch = (data[:, 3] > 0.5).astype(np.uint8)
        wf_sync = (data[:, 4] > 0.5).astype(np.uint8)
        entry_data = dict(**key, t=t, x=x, y=y, touch=touch, wf_sync=wf_sync)

        # detect frames using connected components
        connected_components = measure.label(wf_sync)
        frame_timestamps = []
        exposure = []
        # 1st component are the gaps between frames, 2nd component is the last frame which can be truncated
        frame_ids = np.unique(connected_components)[1:-1] 
        for frame_id in frame_ids:
            frame_indices = np.flatnonzero(connected_components == frame_id)
            current_timestamps = t[frame_indices]
            exposure.append(current_timestamps[-1] - current_timestamps[0])
            frame_timestamps.append(np.median(current_timestamps))
        frame_timestamps = np.array(frame_timestamps)
        exposure = np.array(exposure)

        # estimate framerate and exposure time - assume stable framerate with occasional outlier frames
        exp_median = np.median(exposure)
        exp_std = np.std(exposure)
        diff_timestamps = np.diff(frame_timestamps)
        dt_median = np.median(diff_timestamps)
        dt_std = np.std(diff_timestamps)

        # check if there are any skipped frames, defined as frames with dt > 1.5 * dt_median
        skipped_frames = np.flatnonzero(diff_timestamps > 1.5 * dt_median)
        # check that there are no consecutive skipped frames (i.e. dt > 2.5 * dt_median)
        assert len(np.flatnonzero(diff_timestamps > 2.5 * dt_median)) == 0, "consecutive skipped frames detected"

        # insert skipped timestamps into frame_timestamps
        skipped_timestamps = frame_timestamps[skipped_frames] + dt_median
        frame_timestamps = np.insert(frame_timestamps, skipped_frames+1, skipped_timestamps)

        # make sync entry
        entry_sync = dict(**key, frame_timestamps=frame_timestamps, exp_median=exp_median, exp_std=exp_std, dt_median=dt_median, dt_std=dt_std, n_skipped=len(skipped_frames), skipped_timestamps=skipped_timestamps)
        
        # insert data
        self.insert1(key)
        self.Data.insert1(entry_data)
        self.Sync.insert1(entry_sync)

@schema
class JoystickPresence(dj.Computed):
    definition = """ # cleaned up readouts of joystick in and out times
    -> JoystickReadouts
    ---
    y_baseline                  : float             # baseline y position of joystick
    detection_threshold         : float             # threshold for joystick presence detection
    """
    class Trial(dj.Part):
        definition = """ # cleaned up readouts of joystick in and out times
        -> JoystickPresence
        trial_id                : int               # trial identifier
        ---
        t_joystick_in           : float             # time of joystick in
        t_joystick_out          : float             # time of joystick out
        """

    def plot(self):
        assert len(self) == 1, "plotting only works for one entry at a time"
        key, y_baseline, detection_threshold = self.fetch1("KEY", "y_baseline", "detection_threshold")
        t, y = (JoystickReadouts.Data & key).fetch1("t", "y")
        t_in, t_out = (JoystickPresence.Trial & key).fetch("t_joystick_in", "t_joystick_out", order_by="trial_id")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, y, label="Joystick readout")
        plt.axhline(y_baseline, color='k', ls='--', label="Baseline")
        plt.axhline(detection_threshold, color='r', ls='--', label="Detection threshold")
        plt.vlines(t_in, -2, 2, color='g', ls='--', label="Joystick in")
        plt.vlines(t_out, -2, 2, color='b', ls='--', label="Joystick out")
        plt.xlabel("Time (s)")
        plt.ylabel("Joystick readout")
        plt.legend()
        plt.show()

    def make(self, key):
        # get readouts from joystick sensor
        t, y = (JoystickReadouts.Data & key).fetch1('t', 'y')
        # get trial starts, ends, servo in and out times
        trial_ids, t_start, t_end, t_servo_in, t_servo_out = (JoystickExperiment.Trials & key).fetch('trial_id', 't_start', 't_end', "t_servo_in", "t_servo_out", order_by="trial_id")
        # compute joystick y baseline
        baselines = []
        for t_s, t_e, t_sin, t_sout in zip(t_start, t_end, t_servo_in, t_servo_out):
            b_1 = np.percentile(y[(t>=t_s) & (t<=t_sin)], 95)
            b_2 = np.percentile(y[(t>=t_sout) & (t<=t_e)], 95)
            baselines.append(b_1)
            baselines.append(b_2)
        baselines.sort()
        y_baseline = baselines[4]   # 5th percentile, avoid outliers 
        # compute threshold
        reward_threshold = (JoystickExperiment & key).fetch1("reward_threshold")
        detection_threshold = y_baseline - np.abs(y_baseline-reward_threshold) * 0.2

        # iterate over trials and find threshold crossings
        entries_trials = []
        for trial_id, t_s, t_e in zip(trial_ids, t_start, t_end):
            trial_mask = (t >= t_s) & (t <= t_e)
            y_trial = y[trial_mask]
            t_trial = t[trial_mask]
            # find joystick in and out times
            presence_mask = y_trial < detection_threshold
            # find edges of presence mask
            idx_in, idx_out = np.nonzero(presence_mask)[0][[0, -1]]
            # find in and out times
            t_joystick_in = t_trial[idx_in]
            t_joystick_out = t_trial[idx_out]
            # append entry
            entries_trials.append(dict(**key, trial_id=trial_id, t_joystick_in=t_joystick_in, t_joystick_out=t_joystick_out))
        
        # insert entries
        self.insert1(dict(**key, y_baseline=y_baseline, detection_threshold=detection_threshold))
        self.Trial.insert(entries_trials)
