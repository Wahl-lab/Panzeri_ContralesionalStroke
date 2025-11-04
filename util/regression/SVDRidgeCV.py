from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.metrics import r2_score, make_scorer
import numpy as np

class SVDRidgeCV():
    def __init__(self, u, X, Y, res_out, alphas, n_folds=10, n_jobs=-1, fit_intercept=False):
        """
        Initialize SVDRidgeCV object.
        params:
        u : ndarray
            2D array of shape (n_pixels, n_components)
            Spatial transformation matrix from SVD space to image space
        X : ndarray-like
            Design matrix of shape (n_frames, n_regressors)
        Y : ndarray-like
            Target matrix of shape (n_frames, n_components)
        res_out : tuple
            Output resolution of the spatial maps (height, width)
        alphas : ndarray-like
            Regularization values to search over
        n_folds : int
            Number of cross-validation folds
        n_jobs : int
            Number of parallel jobs to run
            -1 means using all processors
        fit_intercept : bool
            Whether to fit an intercept term
        """
        self._u = u
        self._X = X
        self._Y = Y
        assert len(X) == len(Y), "X and Y must have the same length"
        assert X.dtype == np.float32 and Y.dtype == np.float32, "X and Y must be float32"
        self._h, self._w = res_out
        self._alphas = alphas
        self._n_folds = n_folds
        self._n_jobs = n_jobs
        self._fit_intercept = fit_intercept
        self._cv = KFold(self._n_folds)
        self._scorer = make_scorer(r2_score, multioutput="variance_weighted")
    
    def fit(self):
        """
        Fit the model using cross-validated grid search
        """
        # set up grid search for alpha
        grid_search = GridSearchCV(
            estimator=Ridge(fit_intercept=self._fit_intercept),
            param_grid=dict(alpha=self._alphas),
            scoring=self._scorer,
            n_jobs=self._n_jobs,
            cv=self._cv,
            refit=False
        )
        # run grid search to find best alpha
        grid_search.fit(self._X, self._Y)
        self._best_alpha = grid_search.best_params_["alpha"]
        
        # using best alpha, fit models for each fold
        ridge = Ridge(alpha=self._best_alpha, fit_intercept=self._fit_intercept)
        # fit and evaluate models using cross-validation
        cval =  cross_validate(ridge, self._X, self._Y, cv=self._cv, n_jobs=self._n_jobs, scoring=self._scorer, return_estimator=True, return_indices=True)
        self._r2_scores = cval["test_score"]
        # store estimators
        self._estimators = cval["estimator"]
        # store test and train indices
        self._train_indices = cval["indices"]["train"]
        self._test_indices = cval["indices"]["test"]

    def fit_alpha(self, alpha):
        """
        Fit the model using a specific alpha value
        """
        ridge = Ridge(alpha=alpha, fit_intercept=self._fit_intercept)
        cval =  cross_validate(ridge, self._X, self._Y, cv=self._cv, n_jobs=self._n_jobs, scoring=self._scorer, return_estimator=True, return_indices=True)
        self._r2_scores = cval["test_score"]
        self._estimators = cval["estimator"]
        self._test_indices = cval["indices"]["test"]

    @property
    def r2_scores(self):
        # return variance-weighted r2 scores, of shape (n_folds,)
        return self._r2_scores
    
    def get_r2_maps(self):
        # return spatial r2 maps, of shape (n_folds, h, w)
        maps = np.empty((self._n_folds, self._h, self._w))
        for i, indices in enumerate(self._test_indices):
            X = self._X[indices]
            Y = self._Y[indices]
            Y_pred = self._estimators[i].predict(X)
            maps[i] = self._r2_spatial_score(Y, Y_pred, self._u).reshape((self._h, self._w))
        return maps
    
    @property
    def best_alpha(self):
        return self._best_alpha
    
    @property
    def train_indices(self):
        return self._train_indices
    
    @property
    def test_indices(self):
        return self._test_indices
    
    @property
    def estimators(self):
        return self._estimators

    def get_coefficient_maps(self):
        coef_maps = np.empty((self._n_folds, self._X.ndims, self._h, self._w))
        for i in range(self._n_folds):
            coef = self._estimators[i].coef_
            coef_maps[i] = np.dot(self._u, coef).T.reshape((-1, self._h, self._w))
        return coef_maps
    
    def get_intercept_map(self):
        intercept_map = np.zeros((self._n_folds, self._h, self._w))
        if self._fit_intercept:
            for i in range(self._n_folds):
                intercept = self._estimators[i].intercept_.reshape(-1,1)
                intercept_map[i] = np.dot(self._u, intercept).reshape((self._h, self._w))
        return intercept_map
    
    def get_predictions(self):
        predictions = []
        for i, indices in enumerate(self._test_indices):
            X = self._X[indices]
            Y_pred = self._estimators[i].predict(X)
            predictions.append(Y_pred)
        indices = np.concatenate(self._test_indices)
        predictions = np.vstack(predictions)
        # sort predictions by frame index
        return predictions[np.argsort(indices)]
    
    def get_predictions_subset(self, mask):
        # get predictions for a subset of regressors
        predictions = []
        for i, indices in enumerate(self._test_indices):
            X = self._X[indices][:, mask]   # (n_frames, n_regressors)
            coef = self._estimators[i].coef_[:, mask]  # (n_components, n_regressors) 
            intercept = self._estimators[i].intercept_ # (n_components,) or 0.0
            Y_pred = np.dot(X, coef.T) + intercept     # (n_frames, n_components)
            predictions.append(Y_pred)
        indices = np.concatenate(self._test_indices)
        predictions = np.vstack(predictions)
        # sort predictions by frame index
        return predictions[np.argsort(indices)]

    @staticmethod
    def _r2_spatial_score(y_true, y_pred, u):
        """
        Computes the spatial R^2 maps between test set and predicted values from ridge regression.
        Uses efficient np.einsum function to compute covariance matrices.

        Parameters:
        ----------
        y_true : ndarray
            2D array of shape (n_frames, n_dims)
            Test set of SVD temporal components
        y_pred : ndarray
            2D array of shape (n_frames, n_dims)
            Predicted temporal components from ridge regression model
        u : ndarray
            2D array of shape (n_pixels, n_dims)
            Spatial transformation matrix from SVD space to image space

        Returns:
        ----------
        r2 : ndarray
            1D array of shape (n_pixels,)
        """
        # compute centered arrays
        y_true_centered = y_true - np.mean(y_true, axis=0)
        y_pred_centered = y_pred - np.mean(y_pred, axis=0)
        # compute covariances
        cov_true_pred = np.einsum('ij,kj,kp,ip->i', u, y_true_centered, y_pred_centered, u, optimize="greedy")
        cov_true = np.einsum('ij,kj,kp,ip->i', u, y_true_centered, y_true_centered, u, optimize="greedy")
        cov_pred = np.einsum('ij,kj,kp,ip->i', u, y_pred_centered, y_pred_centered, u, optimize="greedy")
        # output r2 map
        r2 = cov_true_pred ** 2 / (cov_true * cov_pred + 1e-10)
        return r2
    

if __name__ == "__main__":
    from schema import mpanze_widefield as wf, mpanze_paw_tracking as pt, mpanze_exp, mpanze_face_tracking as ft
    import matplotlib.pyplot as plt
    from mpanze_scripts.util.regression.Datasets import RegressionDataset
    key = dict(mouse_id=42, day='2023-02-28', channel='corr')

    # get data
    lag_start = -3
    lag_end = 3
    fs = 20.
    n_lags = int((lag_end - lag_start) * fs + 1)
    lags = np.linspace(lag_start, lag_end, n_lags, endpoint=True)
    alphas = np.logspace(-2, 4, 10)
    n_folds = 10
    Y = (wf.DFFRestingEpochs.Channel() & key).regressor_wf(lags=lags, use_lags=False)
    onsets_ipsi, offsets_ipsi = (pt.MovementClassification() & key).regressor_wf(lags=lags, use_lags=True)
    # plt.figure()
    # plt.plot(onsets_ipsi)
    # plt.show()
    onsets_contra, offsets_contra = (pt.MovementThresholding.Hand() & dict(**key, side='contra')).regressor_wf(lags=lags, use_lags=True)
    onsets_joystick, offsets_joystick = (mpanze_exp.JoystickReadouts() & key).regressor_wf(lags=lags, use_lags=True)
    licking = (ft.DeepLabCut() & key).regressor_wf(lags=lags, use_lags=True)
    rewards, cues = ((mpanze_exp.JoystickExperiment) & key).regressor_wf(lags=lags, use_lags=True)
    X = RegressionDataset(dict(
        onsets_ipsi=onsets_ipsi,
        #offsets_ipsi=offsets_ipsi,
        onsets_contra=onsets_contra,
        #offsets_contra=offsets_contra,
        onsets_joystick=onsets_joystick,
        #offsets_joystick=offsets_joystick,
        licking=licking,
        rewards=rewards,
        cues=cues
        )
    )
    u = (wf.DFFRestingEpochs.Channel() & key).fetch1("u")
    Y = RegressionDataset(dict(svd=Y))

    model = SVDRidgeCV(u, (256, 256), alphas, n_folds=n_folds, n_jobs=-1)

    import time
    t0 = time.perf_counter()
    model.fit(X, Y)
    print(f"Elapsed time: {time.perf_counter() - t0:.3f} s")

    print(model.best_alpha)

    scores = model.r2_scores
    maps = model.r2_maps

    coef = model.coef_

    
    for name in coef.keys():
        plt.figure()
        coef_reg = np.nanmean(coef[name], axis=0)
        plt.imshow(coef_reg[20], cmap='turbo')
        plt.colorbar()
        plt.title(name)

    print(np.mean(scores), np.std(scores))

    plt.figure()
    plt.imshow(np.mean(maps, axis=0), cmap='plasma', vmin=0, vmax=0.8)
    plt.colorbar()
    plt.show()