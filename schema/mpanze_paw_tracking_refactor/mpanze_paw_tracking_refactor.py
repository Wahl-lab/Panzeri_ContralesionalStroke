"""
master script for schema mpanze_paw_tracking_refactor
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# instantiate the schema
schema = dj.schema('mpanze_paw_tracking_refactor', locals(), create_tables=True)

# import all tables from schema into the namespace
from schema.mpanze_paw_tracking_refactor.PawRecording import *
from schema.mpanze_paw_tracking_refactor.DeepLabCut import *
from schema.mpanze_paw_tracking_refactor.Segmentation import *
