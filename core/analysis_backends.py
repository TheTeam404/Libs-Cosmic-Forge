# -*- coding: utf-8 -*-
"""
Wrappers for different analysis backends, primarily Scikit-learn.

Provides functions for common machine learning tasks like scaling, PCA, PLS,
and classification, with error handling and checks for library availability.
"""
import logging
import numpy as np
import pandas as pd
import traceback
from typing import Tuple, Optional, Dict, Any

# --- Scikit-learn Imports ---
# Encapsulated to handle potential ImportError gracefully.
SKLEARN_AVAILABLE = False
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    # from sklearn.cluster import KMeans # Example if adding clustering later
    SKLEARN_AVAILABLE = True
    logging.info("Scikit-learn library found. ML backends enabled.")
except ImportError:
    logging.warning("Scikit-learn library not found. Machine learning analysis features will be disabled. "
                    "Install with 'pip install scikit-learn'.")
    # Define dummy classes/functions to prevent NameErrors if library is missing
    class StandardScaler: pass
    class PCA: pass
    class PLSRegression: pass
    class RandomForestClassifier: pass
    class GradientBoostingClassifier: pass
    class MLPClassifier: pass
    # class KMeans: pass

# --- Backend Check Function ---
def check_sklearn_availability() -> bool:
    """Checks if scikit-learn is installed and available."""
    return SKLEARN_AVAILABLE

# --- Preprocessing ---
def scale_data(X: np.ndarray) -> Optional[np.ndarray]:
    """
    Applies standard scaling (zero mean, unit variance) to the data matrix.

    Args:
        X (np.ndarray): Data matrix (samples x features).

    Returns:
        Optional[np.ndarray]: Scaled data matrix, or None if scaling fails,
                              input is invalid, or scikit-learn is unavailable.

    Warning:
        This function fits the scaler and transforms the entire input X. If X
        represents data that will later be split into training and testing sets,
        applying this function directly to the whole dataset *before* splitting
        will cause data leakage (scaler learns from the test set). For proper
        ML practice, fit the scaler *only* on the training data and use the
        *same* fitted scaler to transform both training and test data separately.
        This function is suitable for exploratory analysis or when applying models
        to the entire dataset at once (e.g., clustering, final PCA).
    """
    if not check_sklearn_availability():
        logging.error("Cannot scale data: Scikit-learn is unavailable.")
        return None
    if X is None or not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
         logging.error("Invalid input data provided for scaling (must be non-empty 2D array).")
         return None
    # Check for NaNs/Infs before scaling
    if not np.all(np.isfinite(X)):
        logging.error("Input data contains NaN or Inf values. Scaling cannot be performed.")
        # Consider imputation as an optional step if needed, but fail by default.
        return None

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logging.info("Data scaling applied successfully (mean=0, std=1).")
        return X_scaled
    except Exception as e:
        logging.error(f"Error during data scaling: {e}", exc_info=True)
        return None

# --- Dimensionality Reduction ---
def run_pca(X: np.ndarray, n_components: int = 3) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Performs Principal Component Analysis (PCA).

    Args:
        X (np.ndarray): Data matrix (samples x features), ideally scaled.
                        Must not contain NaN or Inf values.
        n_components (int): Number of principal components to compute.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]:
            - scores (np.ndarray): PCA scores (samples x n_components).
            - explained_variance_ratio (np.ndarray): Variance explained by each component.
            Returns None if PCA fails, input is invalid, or scikit-learn is unavailable.
    """
    if not check_sklearn_availability():
        logging.error("Cannot perform PCA: Scikit-learn is unavailable.")
        return None
    if X is None or not isinstance(X, np.ndarray) or X.ndim != 2:
        logging.error("Invalid input for PCA (must be 2D array).")
        return None
    # Ensure data is finite before passing to PCA
    if not np.all(np.isfinite(X)):
        logging.error("PCA input data contains NaN or Inf values. Cannot proceed.")
        return None

    n_samples, n_features = X.shape
    if n_samples == 0 or n_features == 0:
        logging.error(f"Cannot perform PCA on empty data (shape: {X.shape}).")
        return None

    # Adjust n_components if it exceeds possible dimensions
    # PCA requires n_components <= min(n_samples, n_features)
    actual_n_components = min(n_components, n_samples, n_features)
    if actual_n_components < 1:
        logging.error(f"Cannot perform PCA: adjusted n_components ({actual_n_components}) must be >= 1 "
                      f"(n_samples={n_samples}, n_features={n_features}).")
        return None
    actual_n_components = int(max(1, actual_n_components)) # Ensure integer and >= 1

    try:
        logging.info(f"Running PCA with n_components={actual_n_components}...")
        pca = PCA(n_components=actual_n_components)
        scores = pca.fit_transform(X)
        variance_explained = pca.explained_variance_ratio_
        logging.info(f"PCA completed. Explained variance ratio: {variance_explained}")
        return scores, variance_explained
    except Exception as e:
        logging.error(f"Error during PCA execution: {e}", exc_info=True)
        return None

# --- Regression ---
def run_pls_regression(X: np.ndarray, y: np.ndarray, n_components: int = 5) -> Optional[Tuple[np.ndarray, float]]:
    """
    Performs Partial Least Squares (PLS) Regression.

    Args:
        X (np.ndarray): Predictor data matrix (samples x features), ideally scaled.
                        Must not contain NaN or Inf values.
        y (np.ndarray): Target variable array (samples x 1 or samples).
                        Must not contain NaN or Inf values.
        n_components (int): Number of PLS components to compute.

    Returns:
        Optional[Tuple[np.ndarray, float]]:
            - y_pred (np.ndarray): Predicted target values (matches original shape of y).
            - r2_score (float): R-squared score of the model fit on the input data.
            Returns None if PLS fails, input is invalid, or scikit-learn is unavailable.
    """
    if not check_sklearn_availability():
        logging.error("Cannot perform PLS: Scikit-learn is unavailable.")
        return None
    if X is None or y is None or not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
         logging.error("Invalid input types for PLS Regression (must be NumPy arrays).")
         return None
    if X.ndim != 2 or y.ndim < 1 or X.shape[0] != y.shape[0]:
         logging.error(f"Input shape mismatch for PLS Regression: X={X.shape}, y={y.shape}.")
         return None
    # Ensure data is finite before passing to PLS
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
         logging.error("PLS input data (X or y) contains NaN or Inf values. Cannot proceed.")
         return None

    n_samples, n_features = X.shape
    original_y_ndim = y.ndim # Store original shape to return consistent predictions

    # Critical check: PLS requires at least 2 samples
    if n_samples < 2:
         logging.error(f"PLS requires at least 2 samples, but found {n_samples}.")
         return None

    # Ensure y is 2D for scikit-learn compatibility, and float type
    y_target = y.reshape(-1, 1) if original_y_ndim == 1 else y
    try:
         y_target_float = y_target.astype(np.float64)
    except Exception as e:
         logging.error(f"Could not convert y target to float64 for PLS: {e}")
         return None

    # Adjust n_components: PLS requires 1 <= n_components <= min(n_samples, n_features)
    # Note: Some implementations might need n_components < n_samples strictly.
    # Let's enforce <= min(n_samples, n_features)
    actual_n_components = min(n_components, n_samples, n_features)
    if actual_n_components < 1:
        logging.error(f"Cannot perform PLS: adjusted n_components ({actual_n_components}) must be >= 1 "
                      f"(n_samples={n_samples}, n_features={n_features}).")
        return None
    actual_n_components = int(max(1, actual_n_components)) # Ensure integer and >= 1

    try:
        logging.info(f"Running PLS Regression with n_components={actual_n_components}...")
        pls = PLSRegression(n_components=actual_n_components)
        pls.fit(X, y_target_float)
        y_pred = pls.predict(X)
        # Calculate R² score on the same data used for fitting
        r2 = pls.score(X, y_target_float)
        logging.info(f"PLS Regression completed. R² = {r2:.4f}")
        # Return flattened predictions if original y was 1D
        return y_pred.flatten() if original_y_ndim == 1 else y_pred, float(r2)
    except Exception as e:
        logging.error(f"Error during PLS Regression execution: {e}", exc_info=True)
        return None

# --- Classification ---
def run_classification(X: np.ndarray, y_labels: np.ndarray, method: str = 'RandomForest', **kwargs) -> Optional[Tuple[np.ndarray, float]]:
    """
    Performs classification using a specified scikit-learn method.

    Trains and predicts on the same input data (X, y_labels).

    Args:
        X (np.ndarray): Data matrix (samples x features), ideally scaled.
                        Must not contain NaN or Inf values.
        y_labels (np.ndarray): True class labels for each sample (1D array).
                               Must not contain NaN or Inf values.
        method (str): Classification method ('RandomForest', 'GBT', 'MLP').
                      Case-sensitive.
        **kwargs: Additional keyword arguments passed to the classifier's constructor.
                  A `random_state=42` is added by default if not provided.

    Returns:
        Optional[Tuple[np.ndarray, float]]:
            - y_pred (np.ndarray): Predicted class labels (1D array).
            - accuracy (float): Accuracy score on the training data.
            Returns None on failure, invalid input, or if scikit-learn is unavailable.

    Warning:
        The reported accuracy is calculated on the **training data only**. This score
        can be misleadingly high and does not reflect the model's ability to
        generalize to new, unseen data. Use cross-validation for a more realistic
        performance estimate.
    """
    if not check_sklearn_availability():
        logging.error("Cannot perform classification: Scikit-learn is unavailable.")
        return None
    if X is None or y_labels is None or not isinstance(X, np.ndarray) or not isinstance(y_labels, np.ndarray):
         logging.error("Invalid input types for classification (must be NumPy arrays).")
         return None
    if X.ndim != 2 or y_labels.ndim != 1 or X.shape[0] != y_labels.shape[0]:
         logging.error(f"Input shape mismatch for classification: X={X.shape}, y_labels={y_labels.shape}.")
         return None
    # Ensure data is finite before passing to classifier
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y_labels)):
         logging.error("Classification input data (X or y_labels) contains NaN or Inf values. Cannot proceed.")
         return None
    if len(np.unique(y_labels)) < 2:
        logging.error("Classification requires at least two distinct classes in y_labels.")
        return None

    logging.info(f"Running classification using {method}...")
    logging.warning("Accuracy reported is on TRAINING data only and may not reflect real-world performance.")
    classifier = None
    # Filter kwargs or pass all? Passing all for simplicity, sklearn usually ignores extras.
    # Add random_state for reproducibility if not provided by caller
    constructor_kwargs = kwargs.copy()
    constructor_kwargs.setdefault('random_state', 42)

    try:
        if method == 'RandomForest':
            constructor_kwargs.setdefault('n_estimators', 100) # Add sensible default
            classifier = RandomForestClassifier(**constructor_kwargs)
        elif method == 'GBT':
            constructor_kwargs.setdefault('n_estimators', 100) # Add sensible defaults
            constructor_kwargs.setdefault('learning_rate', 0.1)
            classifier = GradientBoostingClassifier(**constructor_kwargs)
        elif method == 'MLP':
             constructor_kwargs.setdefault('hidden_layer_sizes', (50, 25)) # Example default
             constructor_kwargs.setdefault('max_iter', 500) # Increase default max_iter
             classifier = MLPClassifier(**constructor_kwargs)
        else:
            logging.error(f"Unsupported classification method: {method}")
            return None

        classifier.fit(X, y_labels)
        y_pred = classifier.predict(X)
        # Calculate accuracy on the training data
        accuracy = np.mean(y_pred == y_labels)
        logging.info(f"{method} classification complete. Training Accuracy = {accuracy:.4f}")
        return y_pred, float(accuracy)

    except Exception as e:
        logging.error(f"Error during {method} classification: {e}", exc_info=True)
        return None

# --- Clustering (Placeholder) ---
# def run_clustering(X: np.ndarray, method: str = 'KMeans', n_clusters: int = 3, **kwargs) -> Optional[np.ndarray]:
#     """ Placeholder for clustering functionality """
#     if not check_sklearn_availability(): logging.error("Cannot cluster: Scikit-learn unavailable."); return None
#     logging.warning(f"Clustering method '{method}' not implemented.")
#     # Example:
#     # if method == 'KMeans':
#     #     try:
#     #         kmeans = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
#     #         labels = kmeans.fit_predict(X)
#     #         return labels
#     #     except Exception as e: logging.error(...); return None
#     return None