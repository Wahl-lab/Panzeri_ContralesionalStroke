"""
Module for the mpanze_glm schema.
Provides access to the tables in the namespace mpanze_glm.
Implements utility functions for interacting with, and populating, the tables.
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# instantiate the schema
schema = dj.schema('mpanze_glm', locals(), create_tables=True)

# import tables
from schema.mpanze_glm.Regressor import Regressor
from schema.mpanze_glm.RidgeModel import RidgeModel
from schema.mpanze_glm.RidgeModelFit2 import RidgeModelFit2, AllenSegmentation
from schema.mpanze_glm.AlignedRidgeModelMap import *
from schema.mpanze_glm.ModelEvaluations import *

###
# Utility functions
###