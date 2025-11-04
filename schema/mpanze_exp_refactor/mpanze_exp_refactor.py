"""
master table for schema mpanze_exp_refactor
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# instantiate the schema
schema = dj.schema('mpanze_exp_refactor', locals(), create_tables=True)

# import all tables from schema into the namespace
from schema.mpanze_exp_refactor.JoystickExperiment import JoystickExperiment
from schema.mpanze_exp_refactor.JoystickReadouts import JoystickReadouts, JoystickPresence
from schema.mpanze_exp_refactor.StrokeDate import StrokeDate, DaysFromStroke, DaysFromStrokeNorm
from schema.mpanze_exp_refactor.Performance import Performance, ExperimentalPhase
from schema.mpanze_exp_refactor.Grouping import Handedness, StrokeGroupName, StrokeGroup, NewGroup
from schema.mpanze_exp_refactor.Histology import *

# utility functions
def populate_all():
    JoystickExperiment.populate(display_progress=True)
    JoystickReadouts.populate(display_progress=True)
    DaysFromStroke.populate(display_progress=True)
    DaysFromStrokeNorm.populate(display_progress=True)
    Performance.populate(display_progress=True)
    ExperimentalPhase.populate(display_progress=True)
