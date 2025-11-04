"""
master script for schema mpanze_widefield_refactor
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# instantiate the schema
schema = dj.schema('mpanze_widefield_refactor', locals(), create_tables=True)

# import all tables from schema into the namespace
from schema.mpanze_widefield_refactor.Hardware import *
from schema.mpanze_widefield_refactor.Registration import *
from schema.mpanze_widefield_refactor.MotionCorrection import *
from schema.mpanze_widefield_refactor.ImageProcessing2 import *
from schema.mpanze_widefield_refactor.Synchronisation import Synchronisation
from schema.mpanze_widefield_refactor.WidefieldSession import WidefieldSession

# utility functions
def import_data():
    WidefieldSession.populate(display_progress=True)
    Synchronisation.populate(display_progress=True, suppress_errors=True)
    ReferenceSession.populate(display_progress=True)

def image_process_high_priority(wf_param_id=0):
    from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp
    keys = (
        exp.DaysFromStrokeNorm
        * exp.ExperimentalPhase
        * exp.StrokeGroup.proj(stroke_group='group')
        * ImageProcessingParameters2
        & "stroke_group!='No Learning'"
        & "stroke_group!='Learning'"
        & "phase!='Learning'"
        & [f"days_from_stroke_norm={d}" for d in [-3,-2,-1,3,7,14,21,28]]
        & "mouse_id>40"
        & f"wf_param_id={wf_param_id}"
    ).fetch('KEY')
    print(len(keys), "sessions to process")
    ImageProcessing2.populate(keys, reserve_jobs=True, order="random", display_progress=True)

def image_process_high_priority_expert(wf_param_id=0):
    from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp
    keys = (
        exp.DaysFromStrokeNorm
        * exp.ExperimentalPhase
        * exp.StrokeGroup.proj(stroke_group='group')
        * ImageProcessingParameters2
        & "stroke_group!='No Learning'"
        & "stroke_group!='Learning'"
        & "phase='Expert'"
        & [f"days_from_stroke_norm={d}" for d in [-3,-2,-1,3,7,14,21,28]]
        & "mouse_id>40"
        & f"wf_param_id={wf_param_id}"
    ).fetch('KEY')
    print(len(keys), "sessions to process")
    ImageProcessing2.populate(keys, reserve_jobs=True, order="random")
