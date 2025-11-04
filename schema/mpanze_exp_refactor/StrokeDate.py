"""
Tables for stroke - timing related information
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema.mpanze_exp_refactor.JoystickExperiment import JoystickExperiment
from schema import common_mice

# other dependencies
import numpy as np

# instantiate the schema
schema = dj.schema('mpanze_exp_refactor', locals(), create_tables=True)

@schema
class StrokeDate(dj.Manual):
    definition = """ # table for storing stroke dates
    -> common_mice.Mouse
    ---
    stroke_date : date          # date of stroke (YYYY-MM-DD)
    """

@schema
class DaysFromStroke(dj.Computed):
    definition = """
    -> JoystickExperiment
    -> StrokeDate
    ---
    days_from_stroke : int      # days from stroke (negative if before stroke)
    """
    def make(self, key):
        # get stroke date
        stroke_date = (StrokeDate & key).fetch1('stroke_date')
        # get session date
        session_date = (JoystickExperiment & key).fetch1('day')
        # compute days from stroke
        days_from_stroke = (session_date - stroke_date).days
        # insert
        self.insert1(dict(**key, days_from_stroke=days_from_stroke))

@schema
class DaysFromStrokeNorm(dj.Computed):
    definition = """
    -> DaysFromStroke
    ---
    days_from_stroke_norm : int      # days from stroke (negative if before stroke)
    """
    def make(self, key):
        # get days from stroke
        days_from_stroke = (DaysFromStroke & key).fetch1('days_from_stroke')
        if days_from_stroke < 0:
            # get ordering from stroke date
            pre_sessions = (DaysFromStroke & "days_from_stroke<0" & dict(mouse_id=key["mouse_id"])).fetch("days_from_stroke", order_by="day DESC")
            # get index of session
            idx = np.where(pre_sessions == days_from_stroke)[0][0] + 1
            new_days_from_stroke = -idx
        else:
            new_days_from_stroke = days_from_stroke
        
        # insert
        self.insert1(dict(**key, days_from_stroke_norm=new_days_from_stroke))
