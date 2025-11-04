"""
Stores ridge model R^2 maps aligned to allen atlas
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import database dependencies
from schema.mpanze_glm.RidgeModelFit import RidgeModelFit
from schema.mpanze_glm.RidgeModelFit2 import RidgeModelFit2
from schema.mpanze_widefield_refactor import mpanze_widefield_refactor as wf
from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp

# other dependencies
import cv2
import numpy as np

# instantiate the schema
schema = dj.schema('mpanze_glm', locals(), create_tables=True)

@schema
class AlignedRidgeModelMap2(dj.Computed):
    definition = """ # stores the results of the Ridge model fit aligned to the allen atlas
    -> RidgeModelFit2
    -> wf.RescaledAllenRegistration2
    -> exp.Handedness
    ---
    r2_map_aligned      : longblob                  # R^2 map averaged over all folds (h,w)
    """
    _key_source = RidgeModelFit2 * wf.RescaledAllenRegistration2

    def make(self, key):
        # fetch affine matrix
        M = (wf.RescaledAllenRegistration2 & key).fetch1('allen_matrix_rescaled')

        handedness = (exp.Handedness & key).fetch1('handedness')

        # process the average map
        r2_map = (RidgeModelFit2 & key).fetch1('r2_map')
        h, w = r2_map.shape
        r2_map_aligned = cv2.warpAffine(r2_map, M, (w, h))
        if handedness == "R":
            r2_map_aligned = np.fliplr(r2_map_aligned)
        entry = dict(**key, r2_map_aligned=r2_map_aligned)

        # insert data
        self.insert1(entry)
