"""
Defines the RidgeModel table for the mpanze_glm schema.
This table specifies the parameters and regressors of ridge regression models
for fitting the task regressors to the widefield imaging data.
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import database dependencies
from schema.mpanze_glm.Regressor import Regressor

# other dependencies
import numpy as np

# instantiate the schema
schema = dj.schema('mpanze_glm', locals(), create_tables=True)

@schema
class RidgeModel(dj.Manual):
    definition = """ # specifies the parameters and regressors of the GLM
    model_id            : int unsigned          # unique model identifier
    ---
    model_name          : varchar(128)           # model short name
    model_description   : varchar(255)          # description of the model
    n_folds             : int                   # number of cross-validation folds
    fit_intercept       : bool                  # whether to fit an intercept
    alphas              : longblob              # alpha values to search for regularization
    use_gmr             : bool                  # whether to use Global Mean Regressed widefield data
    time_insert=CURRENT_TIMESTAMP : timestamp   # automatic timestamp
    """

    class Regressor(dj.Part):
        definition = """ # specifies the regressors used in the GLM
        -> RidgeModel
        -> Regressor
        ---
        shuffle              : bool                             # whether to shuffle the regressor
        lag_mode             : enum('none', 'shift', 'gauss')   # time lags applied to regressor
        lag_frames=NULL      : longblob                         # frames of the time lags (including 0 lag)
        gauss_std=NULL       : float                            # standard deviation of the Gaussian kernel
        use_qr               : bool                             # whether to use QR decomposition for the regressor
        """

    @staticmethod
    def make_fullmodel_0(use_gmr=False, n_folds=5, fit_intercept=True, alphas=np.logspace(-3,7,50,endpoint=True)):
        lag_frames_motor=np.arange(-60, 61)
        lag_frames_events = np.arange(-60, 1)
        model_id = np.max(RidgeModel.fetch('model_id')) + 1 if len(RidgeModel()) > 0 else 0
        master_entry = dict(
            model_id=model_id,
            model_name = "FullModel_0",
            model_description='Full model with only event regressors',
            n_folds=n_folds,
            fit_intercept=fit_intercept,
            alphas=alphas,
            use_gmr=use_gmr
        )
        motor_regressors = ["TaskLimbOnsets", "SupportLimbOnsets"]
        event_regressors = ["ValveOpeningEvents", "CueEvents"]
        motor_entries = [
            dict(
                model_id=model_id,
                reg_name=name,
                shuffle=False,
                lag_mode='shift',
                lag_frames=lag_frames_motor,
                gauss_std=None,  # No Gaussian smoothing for event regressors
                use_qr=False,
            ) for name in motor_regressors
        ]
        event_entries = [
            dict(
                model_id=model_id,
                reg_name=name,
                shuffle=False,
                lag_mode='shift',
                lag_frames=lag_frames_events,
                gauss_std=None,  # No Gaussian smoothing for event regressors
                use_qr=False,
            ) for name in event_regressors
        ]
        RidgeModel.insert1(master_entry)
        RidgeModel.Regressor.insert(event_entries+motor_entries)

    @staticmethod
    def make_fullmodel_0_reduced(regs_to_shuffle, use_gmr=False, n_folds=5, fit_intercept=True, alphas=np.logspace(-3,7,50,endpoint=True)):
        """
        Create a full model with specified regressors to shuffle.
        """
        lag_frames_motor=np.arange(-60, 61)
        lag_frames_events = np.arange(-60, 1)
        model_id = np.max(RidgeModel.fetch('model_id')) + 1 if len(RidgeModel()) > 0 else 0
        master_entry = dict(
            model_id=model_id,
            model_name = "FullModel_0_" + "_".join(regs_to_shuffle),
            model_description='Full model with only event regressors, reduced by shuffling specified regressors',
            n_folds=n_folds,
            fit_intercept=fit_intercept,
            alphas=alphas,
            use_gmr=use_gmr
        )
        motor_regressors = ["TaskLimbOnsets", "SupportLimbOnsets"]
        event_regressors = ["ValveOpeningEvents", "CueEvents"]
        for reg in regs_to_shuffle:
            if reg not in event_regressors + motor_regressors:
                raise ValueError(f"Regressor {reg} not found in the predefined lists.")
        if regs_to_shuffle == ["all"]:
            regs_to_shuffle = event_regressors + motor_regressors
        motor_entries = [
            dict(
                model_id=model_id,
                reg_name=name,
                shuffle=(name in regs_to_shuffle),
                lag_mode='shift',
                lag_frames=lag_frames_motor,
                gauss_std=None,  # No Gaussian smoothing for event regressors
                use_qr=False,
            ) for name in motor_regressors
        ]
        event_entries = [
            dict(
                model_id=model_id,
                reg_name=name,
                shuffle=(name in regs_to_shuffle),
                lag_mode='shift',
                lag_frames=lag_frames_events,
                gauss_std=None,  # No Gaussian smoothing for event regressors
                use_qr=False,
            ) for name in event_regressors
        ]
        RidgeModel.insert1(master_entry)
        RidgeModel.Regressor.insert(event_entries + motor_entries)
