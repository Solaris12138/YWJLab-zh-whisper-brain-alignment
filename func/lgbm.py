import numpy as np
import lightgbm as lgb
import warnings

from joblib import Parallel, delayed


def lgbm_vertex(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    boosting_type="gbdt",
    n_estimators=100,
    learning_rate=0.05,
    reg_lambda=1e-4,
    n_jobs=1,
    early_stopping=10,
    random_state=1016
):
    """
    Using LightGBM as regressor for a single vertex.

    Parameters
    ----------
    X_train : numpy.ndarray
        The features of training set. Shape: (n_samples, n_features)
    y_train : numpy.ndarray
        The target of training set. Shape: (n_samples, )
    
    ...
    
    boosting_type  : str
        'gbdt', Gradient Boosting Decision Tree. 
        'dart', Dropouts meet Multiple Additive Regression Trees. 
        'rf', Random Forest.
    n_estimators : int
        Number of boosted trees to fit.
    learning_rate : float
        Boosting learning rate.
    reg_lambda : flot
        L2 regularization term on weights.
    n_jobs : int
        Number of parallel threads to use for training.
    early_stopping : int
        The number of iterations after which the training procedures would be stopped if the model performance
        did not improve on the validation set.
    
    Returns
    -------
    score : float
        Pearson's r-value between predicted signals and ground truth.
    
    """
    warnings.filterwarnings("ignore")

    # Other parameters are set as default.
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mse",
        boosting_type=boosting_type,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        n_jobs=n_jobs,
        random_state=random_state
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
        lgb.log_evaluation(period=0)
    ]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )

    y_pred = model.predict(X_test)

    if np.std(y_pred) < 1e-9 or np.std(y_test) < 1e-9:
        return 0.0
    
    score = np.corrcoef(y_test, y_pred)[0, 1]
    return score


def lgbm_vertices_serial(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    boosting_type="gbdt",
    n_estimators=100,
    learning_rate=0.05,
    reg_lambda=1e-4,
    n_jobs=1,
    early_stopping=10,
    random_state=1016
):
    """
    Using LightGBM as regressor for multi vertices. (Serial version)

    Parameters
    ----------
    X_train : numpy.ndarray
        The features of training set. Shape: (n_samples, n_features)
    y_train : numpy.ndarray
        The target of training set. Shape: (n_samples, n_vertices)
    
    ...
    
    boosting_type  : str
        'gbdt', Gradient Boosting Decision Tree. 
        'dart', Dropouts meet Multiple Additive Regression Trees. 
        'rf', Random Forest.
    n_estimators : int
        Number of boosted trees to fit.
    learning_rate : float
        Boosting learning rate.
    reg_lambda : flot
        L2 regularization term on weights.
    n_jobs : int
        Number of parallel threads to use for training.
    early_stopping : int
        The number of iterations after which the training procedures would be stopped if the model performance
        did not improve on the validation set.
    
    Returns
    -------
    scores : numpy.ndarray. Shape: (n_vertices, )
        Pearson's r-value between predicted signals and ground truth for all vertices.
    
    """
    V = y_train.shape[-1]

    scores = list()
    for v in range(V):
        score = lgbm_vertex(
            X_train=X_train,
            y_train=y_train[:, v],
            X_val=X_val,
            y_val=y_val[:, v],
            X_test=X_test,
            y_test=y_test[:, v],
            boosting_type=boosting_type,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            early_stopping=early_stopping,
            random_state=random_state
        )
        scores.append(score)

    return np.asarray(scores)


def lgbm_vertices_parallel(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    boosting_type="gbdt",
    n_estimators=100,
    learning_rate=0.05,
    reg_lambda=1e-4,
    n_jobs=1,
    early_stopping=10,
    random_state=1016
):
    """
    Using LightGBM as regressor for multi vertices. (Parallel version)

    Parameters
    ----------
    X_train : numpy.ndarray
        The features of training set. Shape: (n_samples, n_features)
    y_train : numpy.ndarray
        The target of training set. Shape: (n_samples, n_vertices)
    
    ...
    
    boosting_type  : str
        'gbdt', Gradient Boosting Decision Tree. 
        'dart', Dropouts meet Multiple Additive Regression Trees. 
        'rf', Random Forest.
    n_estimators : int
        Number of boosted trees to fit.
    learning_rate : float
        Boosting learning rate.
    reg_lambda : flot
        L2 regularization term on weights.
    n_jobs : int
        Number of parallel threads to use for training.
    early_stopping : int
        The number of iterations after which the training procedures would be stopped if the model performance
        did not improve on the validation set.
    
    Returns
    -------
    scores : numpy.ndarray. Shape: (n_vertices, )
        Pearson's r-value between predicted signals and ground truth for all vertices.
    
    """
    V = y_train.shape[-1]

    scores = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(lgbm_vertex)(
            X_train=X_train,
            y_train=y_train[:, v],
            X_val=X_val,
            y_val=y_val[:, v],
            X_test=X_test,
            y_test=y_test[:, v],
            boosting_type=boosting_type,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            reg_lambda=reg_lambda,
            n_jobs=1,
            early_stopping=early_stopping,
            random_state=random_state
        ) 
        for v in range(V)
    )

    return np.asarray(scores)