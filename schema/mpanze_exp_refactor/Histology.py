"""
Results from histology of tissue
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import table dependencies
from schema import common_mice

# instantiate the schema
schema = dj.schema('mpanze_exp_refactor', locals(), create_tables=True)


@schema
class StrokeVolume(dj.Manual):
    definition = """ # histology of stroke volume
    -> common_mice.Mouse
    ---
    stroke_volume=NULL               : float                                # stroke volume in mm^3
    volume_calculation_method=NULL   : enum('none', 'MRI', 'histology')     # method used to calculate stroke volume
    stroke_vol_notes=''              : varchar(256)                         # additional notes
    """
