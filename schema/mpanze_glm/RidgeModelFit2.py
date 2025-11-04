"""
Table which stores the results of the Ridge model fit.
A k-fold cross-validated ensamble of models is fitted, and individual models are stored in a part table.
Includes model performance evaluations as well as the fitted model coefficients for later reconstruction.
"""

# connect to the database
import login
login.connect()
import datajoint as dj

# import database dependencies
from schema.mpanze_glm.RidgeModel import RidgeModel
from schema.mpanze_glm.Regressor import Regressor
from schema.mpanze_widefield_refactor import mpanze_widefield_refactor as wf
from schema.mpanze_exp_refactor import mpanze_exp_refactor as exp

# other dependencies
import numpy as np
import cv2
from util.regression.SVDRidgeCV import SVDRidgeCV
from util.allen_utils import load_allen

# instantiate the schema
schema = dj.schema('mpanze_glm', locals(), create_tables=True)

@schema
class RidgeModelFit2(dj.Computed):
    definition = """ # stores the results of the Ridge model fit
    -> RidgeModel
    -> wf.ImageProcessing2
    ---
    r2_score            : float                     # R^2 score averaged over all folds
    r2_map              : longblob                  # R^2 map averaged over all folds (h,w)
    alpha_best          : float                     # best alpha value found by cross-validation
    n_regressors        : int                       # number of regressors used in the design matrix (incl. lags)
    n_frames            : int                       # total number of samples used for fitting (number of widefield frames)
    """

    class Fold(dj.Part):
        definition = """ # stores the results of each fold
        -> RidgeModelFit2
        fold_id             : int                   # fold identifier
        ---
        r2_score            : float                 # R^2 score for this fold
        r2_map              : longblob              # R^2 map for this fold (h,w)
        coefficients        : longblob              # fitted model coefficients (n_components, n_regressors)
        intercept           : longblob              # fitted model intercept (n_components,)
        train_indices       : longblob              # indices of training data
        test_indices        : longblob              # indices of test data
        """

    class Regressor(dj.Part):
        definition = """ # stores the regressors used in the model
        -> RidgeModelFit2
        -> Regressor
        ---
        n_dim               : int                   # number of dimensions of the regressor after lagging
        coefficient_mask    : longblob              # mask of size (n_regressors), for selecting coefficients
        """

    def make(self, key):
        # build design matrix
        X, entries_regressors = RidgeModelFit2.build_design_matrix(key)

        # fetch model parameters
        model_params = (RidgeModel & key).fetch1()

        # fetch widefield data
        Y = (wf.ImageProcessing2.TemporalComponents & key).fetch1("svt").T.astype(np.float32) * 100
        u = (wf.ImageProcessing2.SpatialComponents & key).fetch1("u")
        h, w = (wf.ImageProcessingParameters2 & key).fetch_hw()

        # create model
        model = SVDRidgeCV(
            u=u,
            X=X,
            Y=Y,
            res_out=(h, w),
            alphas=model_params["alphas"],
            n_folds=model_params["n_folds"],
            fit_intercept=model_params["fit_intercept"]==1,
            n_jobs=-1
        )

        # fit model
        model.fit()

        # compute scores
        r2_maps = model.get_r2_maps()
        r2_scores = model.r2_scores
        alpha_best = model.best_alpha

        # master entry
        entry_master = dict(
            **key,
            r2_score=np.mean(r2_scores),
            r2_map=np.mean(r2_maps, axis=0),
            alpha_best=alpha_best,
            n_regressors=X.shape[1],
            n_frames=X.shape[0]
        )

        # get fold entries
        entries_folds = []
        for i in range(model_params['n_folds']):
            entry_fold = dict(
                **key,
                fold_id=i,
                r2_score=r2_scores[i],
                r2_map=r2_maps[i],
                coefficients=model.estimators[i].coef_,
                intercept=model.estimators[i].intercept_,
                train_indices=model.train_indices[i],
                test_indices=model.test_indices[i]
            )
            entries_folds.append(entry_fold)

        # insert entries
        self.insert1(entry_master)
        self.Fold.insert(entries_folds)
        self.Regressor.insert(entries_regressors)

    @staticmethod
    def generate_kernel_set(sigma, shifts):
        k_lim = np.max(np.abs(shifts)) + int(sigma + 5)
        k_size = 2 * k_lim + 1
        from scipy.signal.windows import gaussian
        base_kernel = gaussian(k_size, std=int(sigma))
        base_kernel /= np.sum(base_kernel)  # normalize kernel
        kernel_set = []
        for shift in shifts:
            kernel = np.roll(base_kernel, shift)
            kernel_set.append(kernel)
        return np.array(kernel_set).astype(np.float32)
    
    @staticmethod
    def apply_kernel_set(feature, kernel_set):
        convolved_feature = []
        for f in feature.T:
            for kernel in kernel_set:
                f_c = np.convolve(f, kernel, mode='same')
                convolved_feature.append(f_c)
        return np.array(convolved_feature).T.astype(np.float32)

    @staticmethod
    def build_regressor(regressor_dict):
        """
        Takes regressor data and specifications and builds part of the design matrix
        params:
            regressor (dict): dictionary representing a row in the RegressorData * RidgeModel.Regressor table
        """
        data = regressor_dict['reg_data'] # assumes data is already in (n_frames, n_dim) shape

        # shuffle data
        if regressor_dict['shuffle'] == 1:
            rng = np.random.default_rng(42)
            data = rng.permutation(data, axis=0)

        if regressor_dict['lag_mode'] == 'none':
            return data
        elif regressor_dict['lag_mode'] == 'shift':
            lag_frames = regressor_dict['lag_frames']
            return np.hstack([np.roll(data, -i, axis=0) for i in lag_frames])
        elif regressor_dict['lag_mode'] == 'gauss':
            sigma = int(regressor_dict['gauss_std'])
            shifts = regressor_dict['lag_frames']
            kernel_set = RidgeModelFit2.generate_kernel_set(sigma, shifts)
            return RidgeModelFit2.apply_kernel_set(data, kernel_set)
        else:
            raise ValueError(f"Invalid lag_mode '{regressor_dict['lag_mode']}'")
        
    @staticmethod
    def build_design_matrix(key):
        """
        Builds the design matrix of the Ridge model.
        Generates entries for the Regressor part table.
        params:
            key (dict): primary key of the RidgeModelFit table
        """
        from sklearn.preprocessing import RobustScaler, StandardScaler

        regressor_data = []
        regressor_dims = []
        regressor_ids = []
        regressor_names = []
        use_qr = []
        
        entries = []

        # iterate over regressors, sort by QR flag
        regressor_dicts = (RidgeModel.Regressor * Regressor & key).fetch(as_dict=True, order_by='use_qr ASC')
        for i, regressor_dict in enumerate(regressor_dicts):
            reg_data = (Regressor & dict(reg_name=regressor_dict['reg_name'])).load_1(key)
            # build regressor
            regressor_dict['reg_data'] = reg_data
            data = RidgeModelFit2.build_regressor(regressor_dict)
            # if data is empty, e.g. due to an empty event regressor, skip
            if regressor_dict['reg_type'] == 'event' and np.sum(data) == 0:
                print(f"Skipping empty event regressor '{regressor_dict['reg_name']}'")
                continue
            regressor_data.append(data)
            regressor_dims.append(data.shape[1])
            regressor_ids.append(np.ones(data.shape[1], dtype=int) * i)
            regressor_names.append(regressor_dict['reg_name'])
            use_qr.append(regressor_dict['use_qr'])
        
        # build regressor matrix
        X = np.hstack(regressor_data)
        regressor_ids = np.concatenate(regressor_ids)

        # perform QR decomposition if required
        if any(use_qr):
            raise NotImplementedError("QR decomposition is not implemented yet")

        # generate regressor entries
        for i, (data, dim) in enumerate(zip(regressor_data, regressor_dims)):
            entry = dict(
                **key,
                reg_name=regressor_dicts[i]['reg_name'],
                n_dim=dim,
                coefficient_mask=(regressor_ids == i)
            )
            entries.append(entry)

        X = StandardScaler().fit_transform(X).astype(np.float32)  # scale the design matrix
        # X = RobustScaler().fit_transform(X).astype(np.float32)  # scale the design matrix

        return X, entries
    
    def load_predictions(self, assignment_dict, as_stack=False, include_true=False, include_baseline=True):
        key = self.fetch1("KEY")
        keys_folds = (RidgeModelFit2.Fold.proj() & key).fetch("KEY", order_by="fold_id")
        X, _ = RidgeModelFit2.build_design_matrix(key)
        Y_true = (wf.ImageProcessing2.TemporalComponents & key).fetch1("svt").T.astype(np.float32) * 100

        names, masks = [], []
        for name, regressors in assignment_dict.items():
            names.append(name)
            m = (RidgeModelFit2.Regressor & key & [f"reg_name='{reg}'" for reg in regressors]).fetch("coefficient_mask")
            masks.append(np.any(np.stack(m), axis=0))
        
        Y_pred = {name: np.zeros_like(Y_true, dtype=np.float32) for name in names}
        Y_pred["intercept"] = np.zeros_like(Y_true, dtype=np.float32)

        if include_baseline:
            # we include the baseline prediction in the intercept
            Y_subtracted = (wf.BaselineSubtraction2 & key).fetch1("svt_subtracted").T.astype(np.float32) * 100
            Y_pred["intercept"] = Y_subtracted - Y_true

        Y_pred["total"] = np.zeros_like(Y_true, dtype=np.float32)

        for key_fold in keys_folds:
            coefficients, intercept, test_indices = (RidgeModelFit2.Fold.proj("coefficients", "intercept", "test_indices") & key_fold).fetch1("coefficients", "intercept", "test_indices")
            for name, mask in zip(names, masks):
                Y_pred[name][test_indices] += X[test_indices][:,mask] @ coefficients[:,mask].T
            Y_pred["intercept"][test_indices] += intercept
            Y_pred["total"][test_indices] = X[test_indices] @ coefficients.T + Y_pred["intercept"][test_indices]
        
        if include_true:
            Y_pred["true"] = Y_true
        if as_stack:
            u = (wf.ImageProcessing2.SpatialComponents & key).fetch1("u")
            h, w = (wf.ImageProcessingParameters2 & key).fetch_hw()
            for name, y in Y_pred.items():
                Y_pred[name] = (u @ y.T).T.reshape(-1, h, w)
        return Y_pred
    
    def predict(self, as_stack=False):
        key = self.fetch1("KEY")
        keys_folds = (RidgeModelFit2.Fold.proj() & key).fetch("KEY", order_by="fold_id")
        X, _ = RidgeModelFit2.build_design_matrix(key)
        n_frames = self.fetch1("n_frames")
        n_components = (wf.ImageProcessingParameters2 & key).fetch1("n_components_compression")

        Y_pred = np.zeros((n_frames, n_components), dtype=np.float32)
        for key_fold in keys_folds:
            coefficients, intercept, test_indices = (RidgeModelFit2.Fold.proj("coefficients", "intercept", "test_indices") & key_fold).fetch1("coefficients", "intercept", "test_indices")
            Y_pred[test_indices] = X[test_indices] @ coefficients.T + intercept

        if as_stack is True:
            u = (wf.ImageProcessing2.SpatialComponents & key).fetch1("u")
            h, w = (wf.ImageProcessingParameters2 & key).fetch_hw()
            Y_pred = (u @ Y_pred.T).T.reshape(-1, h, w)
            
        return Y_pred
    

@schema
class AllenSegmentation(dj.Computed):
    definition = """ # segment r2 maps using Allen atlas
    -> RidgeModelFit2
    -> wf.RescaledAllenRegistration2
    -> exp.Handedness
    """
    class ROI(dj.Part):
        definition = """ # allen atlas rois
        -> AllenSegmentation
        roi_id          : varchar(64)  # unique id for the roi
        ---
        roi_name        : varchar(64)  # name of the roi
        hemisphere      : enum('ipsi', 'contra')   # hemisphere of the roi, relative to task paw
        r2=NULL         : float        # r2 score of the roi
        n_pixels        : int          # number of valid pixels in the roi
        """

    def make(self, key):
        r2_map = (RidgeModelFit2 & key).fetch1("r2_map")
        h, w = (wf.ImageProcessingParameters2 & key).fetch_hw()
        registered_mask = (wf.ImageProcessing2.RegisteredMask & key).fetch1("mask_dilated")

        # load allen registration matrix
        allen_matrix = (wf.RescaledAllenRegistration2 & key).fetch1("allen_matrix_rescaled")
        M = cv2.invertAffineTransform(allen_matrix)

        # load allen rois
        masks, area_names, edges, mask_total, bregma = load_allen((h,w))

        # register total mask to the imaging session
        mask_total = cv2.warpAffine(mask_total, M, (w, h))
        mask_valid = (registered_mask == 255) & (mask_total > 0)

        # get handedness
        handedness = (exp.Handedness & key).fetch1("handedness")

        # iterate over the rois
        entries_rois = []
        for mask, area_name in zip(masks, area_names):
            name_split = area_name.split("_")
            # register mask to imaging session
            mask_roi = cv2.warpAffine(mask, M, (w, h))
            mask_roi_valid = (mask_roi > 0) & mask_valid
            n_pixels = np.sum(mask_roi_valid)
            if n_pixels > 0:
                r2 = np.nanmean(r2_map[mask_roi_valid])
            else:
                r2 = np.nan

            hemisphere = "ipsi" if name_split[1] == handedness else "contra"

            entry_roi = dict(
                **key,
                roi_id = f"{name_split[0]}_{hemisphere}",
                roi_name = name_split[0],
                hemisphere = hemisphere,
                r2=r2,
                n_pixels = n_pixels
            )
            entries_rois.append(entry_roi)
        # insert the entries
        self.insert1(key)
        self.ROI.insert(entries_rois)
