"""
Metrics for performance evaluation
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema.mpanze_exp_refactor.JoystickExperiment import JoystickExperiment
from schema.mpanze_exp_refactor.StrokeDate import DaysFromStroke, DaysFromStrokeNorm

# instantiate the schema
schema = dj.schema('mpanze_exp_refactor', locals(), create_tables=True)

@schema
class Performance(dj.Computed):
    definition = """ # table for storing performance
    -> JoystickExperiment
    ---
    performance    : float       # performance (fraction correct)
    """
    def make(self, key):
        # get data
        successful = (JoystickExperiment.Trials() & key).fetch("successful")
        # compute performance
        performance = sum(successful) / len(successful)
        # insert
        self.insert1(dict(**key, performance=performance))


@schema
class ExperimentalPhase(dj.Computed):
    definition = """ # table for storing experimental phases
    -> DaysFromStroke
    -> DaysFromStrokeNorm
    -> Performance
    ---
    phase : enum('Learning', 'Expert', 'Early', "Late")     # experimental phase
    """
    def make(self, key):
        days_from_stroke_norm = (DaysFromStrokeNorm & key).fetch1('days_from_stroke_norm')
        days_from_stroke = (DaysFromStroke & key).fetch1('days_from_stroke')
        performance = (Performance & key).fetch1('performance')

        if (performance >= 0.8) & (days_from_stroke <= 0) & (days_from_stroke_norm >= -3):
            phase = 'Expert'
        elif (days_from_stroke <= 0):
            phase = 'Learning'
        elif (days_from_stroke > 0) & (days_from_stroke <= 10):
            phase = 'Early'
        elif (days_from_stroke > 7):
            phase = 'Late'
        else:
            raise Exception("Invalid phase")
        
        self.insert1(dict(**key, phase=phase))
