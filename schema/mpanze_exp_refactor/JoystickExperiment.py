"""
Information about behavior experiment parameters and events
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema import common_mice
from schema import common_exp

# other dependencies
import json
import pandas as pd
from util import pathfinding

# instantiate the schema
schema = dj.schema('mpanze_exp_refactor', locals(), create_tables=True)

@schema
class JoystickExperiment(dj.Imported):
    definition = """ # Info about the joystick experiment parameters and events
    -> common_exp.Session
    ---
    autoreward_probability=NULL  : float         # probability of autoreward on incorrect trials, as a percentage
    baseline                : float         # baseline duration (s)
    max_response_time       : float         # max response time (s)
    intertrial              : float         # post-response time (s)
    cue_frequency           : float         # cue frequency in kHz
    servo_engaged           : bool          # whether the servo is engaged
    punish_threshold        : float         # punishment threshold (V)
    reward_threshold        : float         # reward threshold (V)
    sampling_rate           : float         # sampling rate (Hz)
    servo_in                : float         # position in servo during trial
    servo_out               : float         # position of servo during intertrial
    valve_open_time         : float         # duration of valve opening (s)
    p_params                : varchar(128)  # relative path to .json parameter file
    p_events                : varchar(128)  # relative path to .csv events file
    p_daq                   : varchar(128)  # relative path to .csv daq file
    """
    # include only joytick experiments from batch 7 onwards, as these are formatted as single files
    _key_source = common_exp.Session * common_mice.Mouse.proj(batch='batch') & dict(username='mpanze', task="Joystick_with_cue") & "batch>=7"

    class Trials(dj.Part):
        definition = """ # stores trial timestamps for synchronisation
        -> JoystickExperiment
        trial_id        : int       # trial identifier
        ---
        t_start         : float     # time of start
        t_servo_in      : float     # time of servo in
        t_cue           : float     # time of cue
        t_servo_out     : float     # time that servo comes out (either by timeout, or by successful trial)
        t_end           : float     # time of trial end (conincided with start of next trial)
        successful      : bool      # whether trial was successful
        autoreward      : bool      # whether trial was autorewarded
        """

    def make(self, key):
        # get file paths
        p_params = (common_exp.Session & key).glob("*params.json")[0]
        p_events = (common_exp.Session & key).glob("*events.csv")[0]
        p_daq = (common_exp.Session & key).glob("*daq.csv")[0]

        # read data
        with open(p_params, "r") as f:
            data = json.load(f)
        # make entry
        entry_master = dict(
            **key,
            sampling_rate=data["Sampling rate"],
            autoreward_probability=data["Autoreward probability"],
            valve_open_time=data["Reward time"],
            servo_engaged=1,
            max_response_time=data["Max response time"],
            baseline=data["Baseline"],
            intertrial=data["Intertrial"],
            cue_frequency=data["Cue frequency"]*1000,
            reward_threshold=data["Reward threshold"],
            punish_threshold=data["Punish threshold"],
            servo_in=data["Servo in"],
            servo_out=data["Servo out"],
            p_params=(common_exp.Session & key).get_relative_path(p_params).as_posix(),
            p_events=(common_exp.Session & key).get_relative_path(p_events).as_posix(),
            p_daq=(common_exp.Session & key).get_relative_path(p_daq).as_posix()
            )

        # read events
        df = pd.read_csv(p_events, delimiter=',', header=None)
        t_start = []
        t_servo_in = []
        t_cue = []
        t_servo_out = []
        t_end = []
        successful = []
        autoreward = []

        # TODO: replace spaghetti code
        for i, row in df.iterrows():
            if row[1] == "Baseline":
                if len(t_start) > 0:
                    t_end.append(row[0])
                t_start.append(row[0])
            elif row[1] == "Servo in":
                t_servo_in.append(row[0])
            elif row[1] == "Cue":
                t_cue.append(row[0])
            # deal with different trial outcomes
            elif row[1] == "Fail":
                t_servo_out.append(row[0])
                successful.append(False)
                autoreward.append(False)
            elif row[1] == "Autoreward":
                t_servo_out.append(row[0])
                successful.append(False)
                autoreward.append(True)
            elif row[1] == "Reward":
                t_servo_out.append(row[0])
                successful.append(True)
                autoreward.append(False)
            elif row[1] == "End":
                t_end.append(row[0])
            else:
                raise Exception("invalid row")
        
        # make trial entries
        entries_trials = []
        for i in range(len(t_start)):
            entry = dict(
                **key,
                trial_id=i,
                t_start=t_start[i],
                t_servo_in=t_servo_in[i],
                t_cue=t_cue[i],
                t_servo_out=t_servo_out[i],
                t_end=t_end[i],
                successful=successful[i],
                autoreward=autoreward[i]
            )
            entries_trials.append(entry)

        # insert data
        self.insert1(entry_master)
        self.Trials.insert(entries_trials)

    def get_path(self, file, as_posix=False):
        """
        get absolute path to one of the files associated with the experiment
        params:
            as_posix: bool, if True return path as string (posix format), otherwise as Path object
        """
        assert len(self) == 1, "only one entry allowed"
        p = pathfinding.get_absolute_paths(self, file)
        assert len(p) == 1, "only one file allowed"
        return p[0].as_posix() if as_posix else p[0] 
