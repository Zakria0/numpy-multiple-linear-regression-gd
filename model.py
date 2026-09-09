"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """
    # TODO: Return (X, y) under one shared seeded row permutation
    n, d = X.shape
    np.random.seed(seed)
    p = np.random.permutation(n)

    return X[p, :], y[p]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    return (
        X[:n_train], y[:n_train],
        X[n_train: n_train + n_val], y[n_train: n_train + n_val],
        X[n_train + n_val:], y[n_train + n_val:]
    )

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    n, d = X.shape
    mu = np.zeros(d)
    sigma = np.zeros(d)

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    
    return mu, np.where(sigma == 0, 1.0, sigma)

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    n, d = X.shape

    return (X - mean) / std

# Step 5 - add_bias_column
def add_bias_column(X):
    n = X.shape[0]

    return np.column_stack((np.ones(n), X))

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    X_std = standardize_features(X, mean, std)
    design_mat = add_bias_column(X_std)

    return design_mat

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    return X @ weights

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    n = X.shape[0]
    return 2 / n * X.T @ (y_pred - y_true)

# Step 10 - normal_equation
def normal_equation(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    return np.random.normal(0, 0.01, n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    return weights - lr * mse_gradient(X, y, X @ weights)

# Step 13 - epoch_train_val_losses
def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights.

    Args:
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        weights: Weight vector of shape (d_in,).

    Returns:
        (train_loss, val_loss) as plain floats.
    """
    return (
        mse_loss(y_train, X_train @ weights),
        mse_loss(y_val, X_val @ weights)
    )

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    if val_loss < best_val_loss:
        best_weights = weights.copy()
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1
    
    return best_val_loss, wait, best_weights, wait >= patience

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    weights = initialize_weights(n_features, seed)
    return {
        'weights': weights,
        'best_weights': weights.copy(),
        'best_val_loss': np.inf,
        'wait': 0,
        'train_losses': [],
        'val_losses': [],
        'stopped': False
        }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state.

    Args:
        state: Dict with keys weights, best_weights, best_val_loss, wait,
            stopped, train_losses, val_losses.
        X_train: Training design matrix of shape (n_tr, d_in).
        y_train: Training targets of shape (n_tr,).
        X_val: Validation design matrix of shape (n_va, d_in).
        y_val: Validation targets of shape (n_va,).
        lr: Learning rate (float).
        patience: Early-stopping patience (int).

    Returns:
        Updated state dict.
    """
    state['weights'] = gd_step(X_train, y_train, state['weights'], lr)
    mse_loss_train, mse_loss_val = epoch_train_val_losses(X_train, y_train, X_val, y_val, state['weights'])

    state['train_losses'].append(mse_loss_train)
    state['val_losses'].append(mse_loss_val)

    s = update_early_stop_state(mse_loss_val, state['best_val_loss'], state['wait'], state['weights'], state['best_weights'], patience)

    state['best_val_loss'] = s[0]
    state['wait'] = s[1]
    state['best_weights'] = s[2]
    state['stopped'] = s[3]

    return state

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

