# -*- coding: utf-8 -*-
"""
Core signal processing functions for LIBS spectra.

Includes baseline correction, smoothing, wavelet denoising, peak profile functions,
and noise analysis capabilities.
"""

import logging
import numpy as np
import warnings # Import warnings module
from typing import Tuple, Optional, List, Any # Removed Union

# --- SciPy Import Handling ---
SCIPY_AVAILABLE = False
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    logging.warning("SciPy not found. Savitzky-Golay smoothing will be unavailable. Install with 'pip install scipy'.")
    def savgol_filter(*args, **kwargs): raise ImportError("SciPy required for Savitzky-Golay smoothing but is not installed.")

# --- PyWavelets Import Handling ---
PYWAVELETS_AVAILABLE = False
try:
    import pywt # Default import name for PyWavelets
    PYWAVELETS_AVAILABLE = True
except ImportError:
    logging.warning("PyWavelets not found. Wavelet denoising will be unavailable. Install with 'pip install PyWavelets'.")
    # Dummy class to avoid NameErrors if pywt is used directly elsewhere, but raise error on call
    class pywt:
        @staticmethod
        def wavedec(*args, **kwargs): raise ImportError("PyWavelets required but not installed.")
        @staticmethod
        def threshold(*args, **kwargs): raise ImportError("PyWavelets required but not installed.")
        @staticmethod
        def waverec(*args, **kwargs): raise ImportError("PyWavelets required but not installed.")
        @staticmethod
        def dwt_max_level(*args, **kwargs): raise ImportError("PyWavelets required but not installed.")
        @staticmethod
        def Wavelet(*args, **kwargs): raise ImportError("PyWavelets required but not installed.")

# --- Utility Import ---
try:
    # Assuming utils.helpers exists in the project structure relative to core
    from utils.helpers import ensure_odd
except ImportError:
    logging.warning("utils.helpers.ensure_odd not found. Using basic implementation.")
    def ensure_odd(n: Any) -> int:
        """Ensures an integer is odd."""
        try:
            n_int = int(n)
            return n_int if n_int % 2 != 0 else n_int + 1
        except (ValueError, TypeError):
             logging.warning(f"Could not convert {n} to int in ensure_odd, returning 3.")
             return 3

# --- Constants ---
FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0)) # Approx 2.35482
EPSILON = 1e-9 # Small epsilon for numerical stability (e.g., avoid division by zero)
MAD_NORMALIZATION_CONST = 0.6745 # Approx normalization MAD -> sigma for Gaussian noise


# --- Utility Functions ---

def _interpolate_finite(y: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Internal helper to interpolate NaN/Inf values using linear interpolation.

    Args:
        y (np.ndarray): Input array, potentially containing NaN/Inf.

    Returns:
        Tuple[np.ndarray, bool]:
            - y_interp (np.ndarray): Array with NaN/Inf values interpolated (or original if none/failed).
            - success (bool): True if interpolation was successful or not needed, False otherwise.
    """
    y_interp = y.copy() # Work on a copy
    finite_mask = np.isfinite(y_interp)

    if np.all(finite_mask):
        return y_interp, True # No interpolation needed

    logging.warning(f"Found {np.sum(~finite_mask)} non-finite values. Attempting linear interpolation.")
    finite_indices = np.flatnonzero(finite_mask)
    nan_indices = np.flatnonzero(~finite_mask)

    # Check if interpolation is possible
    if len(finite_indices) < 2:
        logging.error("Cannot interpolate NaNs/Infs: Fewer than 2 finite data points found.")
        # Cannot interpolate, return original array but signal failure
        return y, False

    # Perform interpolation
    try:
        y_interp[nan_indices] = np.interp(nan_indices, finite_indices, y_interp[finite_indices])
    except Exception as e:
        logging.error(f"Error during np.interp for NaN/Inf interpolation: {e}", exc_info=True)
        # Interpolation failed, return original array and signal failure
        return y, False

    # Final check if interpolation successfully removed all non-finite values
    if not np.all(np.isfinite(y_interp)):
        # This shouldn't happen with linear interpolation if there were >= 2 finite points, but safety check
        logging.error("Interpolation completed, but non-finite values still remain. This indicates an unexpected issue.")
        return y, False # Signal failure if still contains non-finite values

    logging.debug(f"Successfully interpolated {len(nan_indices)} NaN/Inf values.")
    return y_interp, True


# --- Profile Functions ---
# Removed sigma/gamma <= EPSILON checks, rely on fitter bounds instead. Added EPSILON in sqrt.
def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Gaussian profile function (amplitude, center, standard deviation sigma)."""
    # Sigma must be > 0, handled by fitter bounds.
    return amplitude * np.exp(-((x - center)**2) / (2 * (sigma**2 + EPSILON)))

def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    """Lorentzian profile function (amplitude, center, HWHM gamma)."""
    # Gamma (HWHM) must be > 0, handled by fitter bounds.
    return amplitude * ((gamma**2 + EPSILON) / ((x - center)**2 + (gamma**2 + EPSILON)))

def pseudo_voigt(
    x: np.ndarray, amplitude: float, center: float, sigma: float, eta: float
) -> np.ndarray:
    """Pseudo-Voigt profile using Gaussian sigma and mixing parameter eta (0=Gauss, 1=Lorentz)."""
    # Sigma > 0 and 0 <= eta <= 1 handled by fitter bounds.
    eta_bounded = np.clip(eta, 0.0, 1.0) # Ensure eta is valid
    # Avoid FWHM calculation here; calculate gamma based on sigma for consistency
    gamma = max(EPSILON, sigma * FWHM_GAUSS_FACTOR / 2.0)
    gauss_part = (1.0 - eta_bounded) * np.exp(-((x - center)**2) / (2 * (sigma**2 + EPSILON)))
    loren_part = eta_bounded * ((gamma**2 + EPSILON) / ((x - center)**2 + (gamma**2 + EPSILON)))
    return amplitude * (gauss_part + loren_part)


# --- Baseline Correction Algorithms ---

def baseline_poly(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    order: int = 3,
    percentile: float = 10.0
    # Removed max_iterations, tolerance - performing single pass fit
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates and subtracts a polynomial baseline fitted to points below a percentile threshold.

    Performs a single-pass fit (no iterative refinement). Handles NaNs/Infs via interpolation.

    Args:
        wavelengths (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array.
        order (int): Order of the polynomial to fit (>= 0).
        percentile (float): Percentile threshold (0 < percentile <= 100) to select baseline points.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - corrected_intensity: Intensity with baseline subtracted.
            - baseline: Calculated baseline array (NaN where input was NaN/Inf).
            Returns (original_intensity, zeros_baseline) if baseline cannot be calculated.
    """
    if not isinstance(wavelengths, np.ndarray) or not isinstance(intensity, np.ndarray):
        raise TypeError("Wavelengths and intensity must be NumPy arrays.")
    if wavelengths.shape != intensity.shape:
        raise ValueError(f"Wavelength ({wavelengths.shape}) and Intensity ({intensity.shape}) arrays must have the same shape.")

    n_points = len(wavelengths)
    min_points_required = order + 1 # Points needed for polyfit

    # --- Input Validation ---
    if order < 0:
        logging.error("Polynomial baseline order cannot be negative. Skipping baseline.")
        return intensity, np.zeros_like(intensity)
    if n_points < min_points_required:
        logging.error(f"Polynomial baseline skipped: Need at least {min_points_required} points for order {order}, but got {n_points}.")
        return intensity, np.zeros_like(intensity)
    if not (0 < percentile <= 100):
        original_percentile = percentile
        percentile = 10.0 # Use a sensible default
        logging.warning(f"Invalid percentile ({original_percentile}). Using default: {percentile}%.")

    # --- Handle NaNs/Infs ---
    intensity_processed, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
        logging.error("Failed to handle NaNs/Infs via interpolation. Skipping polynomial baseline.")
        return intensity, np.zeros_like(intensity) # Return original data

    # --- Select Points and Fit (Single Pass) ---
    baseline = np.zeros_like(intensity_processed)
    try:
        # Calculate threshold based on the (potentially interpolated) processed intensity
        threshold = np.percentile(intensity_processed[np.isfinite(intensity_processed)], percentile)
        # Mask includes only points below threshold AND originally finite points
        finite_mask = np.isfinite(intensity) # Use original finiteness
        mask = (intensity_processed <= threshold) & finite_mask
        n_masked_pts = np.sum(mask)

        # Check if enough points selected
        if n_masked_pts < min_points_required:
            logging.warning(f"Only {n_masked_pts} finite points below {percentile:.1f}% percentile "
                            f"(threshold={threshold:.2f}). Need {min_points_required}. "
                            "Using lowest intensity points as fallback.")
            # Fallback: Use lowest finite points
            finite_indices = np.where(finite_mask)[0]
            if len(finite_indices) < min_points_required:
                logging.error(f"Fallback failed: Only {len(finite_indices)} finite points available. Cannot fit polynomial.")
                return intensity, np.zeros_like(intensity)
            # Sort finite points by intensity and take the lowest required number
            sorted_finite_indices = finite_indices[np.argsort(intensity_processed[finite_indices])]
            fallback_indices = sorted_finite_indices[:min_points_required]
            mask = np.zeros_like(intensity_processed, dtype=bool)
            mask[fallback_indices] = True
            n_masked_pts = np.sum(mask) # Should be min_points_required
            logging.debug(f"Fallback selected {n_masked_pts} lowest finite points.")

        # Perform polynomial fit on selected points
        x_masked = wavelengths[mask]
        y_masked = intensity_processed[mask] # Fit to the (potentially interpolated) intensity

        # Final check for finite values before polyfit (should be redundant now)
        if not (np.all(np.isfinite(x_masked)) and np.all(np.isfinite(y_masked))):
             logging.error("Non-finite values detected in data selected for polyfit after masking/fallback. Skipping.")
             return intensity, np.zeros_like(intensity)

        coeffs = np.polyfit(x_masked, y_masked, order)
        baseline = np.polyval(coeffs, wavelengths)

    except (np.linalg.LinAlgError, ValueError) as e:
        logging.error(f"Polynomial baseline fitting failed: {e}", exc_info=True)
        return intensity, np.zeros_like(intensity) # Return original on fit error
    except Exception as e:
        logging.error(f"Unexpected error during polynomial baseline calculation: {e}", exc_info=True)
        return intensity, np.zeros_like(intensity)

    # --- Apply Correction and Return ---
    # Subtract baseline from the interpolated data
    corrected_intensity = intensity_processed - baseline
    # Restore original NaNs/Infs in the output
    corrected_intensity[~finite_mask] = intensity[~finite_mask]
    baseline[~finite_mask] = np.nan # Baseline is undefined where original was NaN/Inf

    logging.info(f"Polynomial baseline applied (order={order}, percentile={percentile:.1f}%).")
    return corrected_intensity, baseline


def baseline_snip(
    wavelengths: np.ndarray, # Kept for API consistency, but not used by SNIP
    intensity: np.ndarray,
    max_iterations: int = 100,
    increasing_window: bool = True # Default from config usually
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the baseline using the Sensitive Nonlinear Iterative Peak (SNIP) algorithm.

    Handles NaNs/Infs via interpolation.

    Args:
        wavelengths (np.ndarray): Wavelength array (unused by SNIP algorithm itself).
        intensity (np.ndarray): Intensity array.
        max_iterations (int): Number of iterations (clipping window size increases up to this).
        increasing_window (bool): If True (default), window size increases from 1 to max_iterations.
                                  If False, uses a fixed window size of max_iterations.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - corrected_intensity: Intensity with baseline subtracted.
            - baseline: Calculated baseline array (NaN where input was NaN/Inf).
            Returns (original_intensity, zeros_baseline) if baseline cannot be calculated.
    """
    if not isinstance(intensity, np.ndarray):
        raise TypeError("Intensity must be a NumPy array.")

    n_points = len(intensity)
    if n_points < 3:
        logging.warning(f"SNIP requires >= 3 points, got {n_points}. Skipping baseline.")
        return intensity, np.zeros_like(intensity)
    if max_iterations < 1:
        logging.warning(f"SNIP max_iterations ({max_iterations}) must be >= 1. Setting to 1.")
        max_iterations = 1

    # --- Handle NaNs/Infs ---
    intensity_processed, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
        logging.error("Failed to handle NaNs/Infs via interpolation. Skipping SNIP baseline.")
        return intensity, np.zeros_like(intensity) # Return original data

    # --- Run SNIP ---
    baseline = intensity_processed.copy() # Start baseline estimate with the data
    finite_mask = np.isfinite(intensity) # Use original finiteness

    logging.info(f"Starting SNIP baseline estimation (max_iter={max_iterations}, increasing_window={increasing_window}).")
    try:
        for k in range(1, max_iterations + 1):
            # Determine window size for this iteration
            window_size = k if increasing_window else max_iterations

            # Check if window size exceeds array limits for slicing
            # Need 2*window_size for accessing prev_baseline
            # Need window_size for accessing baseline itself
            if (2 * window_size) >= n_points or window_size >= n_points:
                 logging.debug(f"SNIP iteration {k}: Window size ({window_size}) too large for data length ({n_points}). Stopping.")
                 break # Stop if window is too large for array access

            # Store previous iteration's baseline
            prev_baseline = baseline.copy()

            # Calculate average of neighbors at distance 'window_size'
            # Slice carefully to avoid index errors
            start_avg = 0
            end_avg_left = n_points - (2 * window_size)
            start_avg_right = 2 * window_size
            end_avg = n_points

            # Ensure slices are valid
            if end_avg_left <= start_avg or start_avg_right >= end_avg:
                logging.warning(f"SNIP iteration {k}: Window size ({window_size}) too large for neighbor averaging. Stopping.")
                break

            avg_neighbors = (prev_baseline[start_avg : end_avg_left] +
                             prev_baseline[start_avg_right : end_avg]) / 2.0

            # Compare points to the average of their neighbors
            points_to_clip_slice = baseline[window_size : -window_size]
            # Ensure avg_neighbors has the same length as points_to_clip_slice
            if len(avg_neighbors) != len(points_to_clip_slice):
                 logging.error(f"SNIP iteration {k}: Slice length mismatch ({len(points_to_clip_slice)} vs {len(avg_neighbors)}). Stopping.")
                 break

            points_to_clip_mask = points_to_clip_slice > avg_neighbors

            # Apply clipping where intensity is greater than neighbors' average
            # Slice baseline again for assignment
            baseline[window_size : -window_size][points_to_clip_mask] = avg_neighbors[points_to_clip_mask]

        # Ensure baseline doesn't go above the processed (interpolated) intensity
        baseline = np.minimum(baseline, intensity_processed)

    except Exception as e:
        logging.error(f"Error during SNIP baseline correction iterations: {e}", exc_info=True)
        return intensity, np.zeros_like(intensity) # Return original on error

    # --- Apply Correction and Return ---
    corrected_intensity = intensity_processed - baseline
    # Restore original NaNs/Infs in the output
    corrected_intensity[~finite_mask] = intensity[~finite_mask]
    baseline[~finite_mask] = np.nan # Baseline undefined where original was NaN/Inf

    logging.info("SNIP baseline estimation complete.")
    return corrected_intensity, baseline


# --- Smoothing Algorithms ---

def smooth_savitzky_golay(
    intensity: np.ndarray,
    smoothing_window: int = 11,
    smoothing_polyorder: int = 3
) -> np.ndarray:
    """
    Applies Savitzky-Golay smoothing to the intensity data.

    Handles NaNs/Infs via interpolation before smoothing.
    Validates input parameters strictly and returns original data if invalid.

    Args:
        intensity (np.ndarray): Intensity array to smooth.
        smoothing_window (int): Window length for the filter (odd integer >= 3).
        smoothing_polyorder (int): Polynomial order for the filter (>= 0, < window length).

    Returns:
        np.ndarray: The smoothed intensity array, or the original array if smoothing fails
                   or parameters are invalid.
    """
    if not SCIPY_AVAILABLE:
        logging.error("Cannot smooth: SciPy library is unavailable.")
        return intensity
    if not isinstance(intensity, np.ndarray):
        raise TypeError("Intensity must be a NumPy array.")

    n_points = len(intensity)
    if n_points < 3:
        logging.warning(f"Data too short ({n_points} points) for Sav-Gol smoothing. Returning original.")
        return intensity

    # --- Strict Parameter Validation ---
    try:
        # Ensure window is integer and make odd >= 3
        wl = ensure_odd(int(smoothing_window))
        if wl < 3:
            logging.error(f"Invalid Sav-Gol window ({smoothing_window}). Must result in odd integer >= 3. Cannot smooth.")
            return intensity
        # Ensure polyorder is integer >= 0
        po = int(smoothing_polyorder)
        if po < 0:
            logging.error(f"Invalid Sav-Gol polyorder ({smoothing_polyorder}). Must be >= 0. Cannot smooth.")
            return intensity
        # Check window > polyorder
        if wl <= po:
             logging.error(f"Invalid Sav-Gol parameters: window_length ({wl}) must be greater than polyorder ({po}). Cannot smooth.")
             return intensity
        # Check window <= n_points
        if wl > n_points:
             logging.error(f"Invalid Sav-Gol window_length ({wl}): Cannot exceed data length ({n_points}). Cannot smooth.")
             return intensity
    except (ValueError, TypeError) as e:
        logging.error(f"Invalid Sav-Gol parameters (window={smoothing_window}, order={smoothing_polyorder}): {e}. Cannot smooth.")
        return intensity

    # --- Handle NaNs/Infs ---
    intensity_processed, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
        logging.error("Failed to handle NaNs/Infs prior to smoothing. Returning original.")
        return intensity
    finite_mask = np.isfinite(intensity) # Keep track of original non-finite positions

    # --- Apply Filter ---
    try:
        # Use validated wl, po parameters
        smoothed_intensity = savgol_filter(intensity_processed, window_length=wl, polyorder=po)
        # Restore original NaNs/Infs in the output
        smoothed_intensity[~finite_mask] = intensity[~finite_mask]
        logging.info(f"Applied Savitzky-Golay smoothing (window={wl}, order={po}).")
        return smoothed_intensity
    except ValueError as e:
        # This might catch issues if internal checks fail despite initial validation
        logging.error(f"Error applying Savitzky-Golay filter: {e}. Returning original.", exc_info=True)
        return intensity
    except Exception as e:
        logging.error(f"Unexpected error during Savitzky-Golay smoothing: {e}", exc_info=True)
        return intensity


# --- Denoising Algorithms ---

def denoise_wavelet(
    intensity: np.ndarray,
    wavelet_type: str = 'db8',
    level: Optional[int] = None,
    mode: str = 'soft',          # Thresholding mode ('soft' or 'hard')
    threshold_sigma_factor: float = 3.0 # Factor times MAD for threshold
) -> np.ndarray:
    """
    Applies wavelet denoising to the intensity data using MAD thresholding.

    Handles NaNs/Infs via linear interpolation before denoising.

    Args:
        intensity (np.ndarray): Intensity array to denoise.
        wavelet_type (str): Type of wavelet (e.g., 'db4', 'sym8', 'coif5').
        level (Optional[int]): Decomposition level. If None, automatically estimated.
        mode (str): Thresholding mode ('soft' or 'hard').
        threshold_sigma_factor (float): Factor to multiply MAD by for threshold value (> 0).

    Returns:
        np.ndarray: The denoised intensity array, or the original array if denoising fails.
    """
    if not PYWAVELETS_AVAILABLE:
        logging.error("Cannot denoise: PyWavelets library is unavailable.")
        return intensity # Return original data

    if not isinstance(intensity, np.ndarray):
        raise TypeError("Intensity must be a NumPy array.")

    n_points = len(intensity)
    if n_points < 2: # Need at least 2 points for decomposition
        logging.warning(f"Data too short ({n_points} points) for wavelet denoising. Returning original.")
        return intensity

    # --- Handle NaNs/Infs using Interpolation ---
    intensity_processed, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
        logging.error("Failed to handle NaNs/Infs via interpolation. Skipping wavelet denoising.")
        return intensity
    finite_mask = np.isfinite(intensity) # Keep track of original non-finite positions

    # --- Parameter Validation ---
    try:
        wavelet = pywt.Wavelet(wavelet_type)
    except ValueError:
        logging.error(f"Invalid wavelet type '{wavelet_type}'. Available families: {pywt.families()}. Skipping denoising.")
        return intensity

    # Determine decomposition level
    try:
        max_level = pywt.dwt_max_level(n_points, wavelet.dec_len)
        if max_level < 1:
             logging.error(f"Data length {n_points} too short for even 1 level of decomposition with wavelet '{wavelet_type}'. Skipping.")
             return intensity
    except Exception as e_level:
         logging.error(f"Could not determine max wavelet level: {e_level}. Skipping.")
         return intensity

    if level is None:
        level = max_level # Default to max level if not specified
        logging.debug(f"Wavelet level automatically determined: {level}")
    else:
        try: level = int(level)
        except (ValueError, TypeError): logging.warning(f"Invalid level '{level}', using max {max_level}."); level = max_level
        if level < 1: logging.warning(f"Wavelet level ({level}) must be >= 1. Using 1."); level = 1
        elif level > max_level: logging.warning(f"Requested level ({level}) > max level ({max_level}). Using max level."); level = max_level

    # Validate mode
    if mode not in ['soft', 'hard']:
        logging.warning(f"Invalid thresholding mode '{mode}'. Using default 'soft'.")
        mode = 'soft'

    # Validate sigma factor
    if threshold_sigma_factor <= 0:
        logging.warning(f"Threshold sigma factor ({threshold_sigma_factor}) must be positive. Using default 3.0.")
        threshold_sigma_factor = 3.0

    # --- Perform Denoising ---
    try:
        # --- Wavelet Decomposition ---
        # Use padding mode 'symmetric' (or 'reflect') to handle boundaries gracefully
        coeffs = pywt.wavedec(intensity_processed, wavelet, level=level, mode='symmetric')
        # coeffs structure: [cA_n, cD_n, cD_n-1, ..., cD_1]

        coeffs_thresholded = [coeffs[0]] # Keep approximation coeffs (cA_n) unchanged

        # --- Threshold Detail Coefficients ---
        # Iterate through detail levels from coarsest (cD_n) to finest (cD_1)
        for i in range(level, 0, -1):
            detail_coeffs = coeffs[level - i + 1] # Get cD_i array

            # Estimate noise standard deviation (sigma) using MAD of this level's coeffs
            # Note: Common practice is to estimate sigma from the *finest* level (cD_1) only.
            # Let's stick to that for consistency unless level-dependent thresholding is intended.
            # --- Noise Estimation (MAD from finest detail coefficients cD1) ---
            if i == 1: # Only estimate from finest level (cD1)
                coeffs_cd1 = detail_coeffs
                if len(coeffs_cd1) == 0:
                     logging.warning("No detail coefficients found at level 1. Cannot estimate noise. Skipping thresholding for all levels.")
                     # If noise cannot be estimated, skip thresholding for all levels
                     coeffs_thresholded = coeffs[1:] # Append all original detail coeffs
                     break # Exit the loop

                median_cd1 = np.median(coeffs_cd1)
                mad = np.median(np.abs(coeffs_cd1 - median_cd1))

                if mad < EPSILON:
                    # If MAD is zero, noise estimation is unreliable. Skip thresholding.
                    logging.warning(f"MAD of detail coefficients (cD1) is near zero ({mad:.2e}). Noise estimation unreliable. Skipping thresholding for all levels.")
                    coeffs_thresholded = coeffs[1:] # Use original detail coeffs
                    break # Exit the loop
                else:
                    # Estimate noise standard deviation (sigma) using MAD
                    sigma = mad / MAD_NORMALIZATION_CONST
                    # Calculate threshold based on this single sigma estimate
                    threshold_value = threshold_sigma_factor * sigma
                    logging.info(f"Applying Wavelet Denoising: Wavelet='{wavelet_type}', Level={level}, Mode='{mode}', "
                                 f"Est. Noise (Sigma from cD1)≈{sigma:.3f}, Threshold={threshold_value:.3f}")
            # --- End Noise Estimation ---

            # Apply thresholding (if threshold_value was calculated successfully)
            try:
                thresholded_detail_coeffs = pywt.threshold(detail_coeffs, value=threshold_value, mode=mode)
                coeffs_thresholded.append(thresholded_detail_coeffs)
            except NameError: # Handle case where threshold_value wasn't set (e.g., cD1 empty)
                logging.debug(f"Skipping threshold for level cD{i} as noise estimation failed.")
                coeffs_thresholded.append(detail_coeffs) # Append original coeffs
            except Exception as e_thresh:
                 logging.error(f"Error during pywt.threshold for level cD{i}: {e_thresh}", exc_info=True)
                 coeffs_thresholded.append(detail_coeffs) # Append original on error


        # --- Wavelet Reconstruction ---
        denoised_intensity = pywt.waverec(coeffs_thresholded, wavelet, mode='symmetric')

        # Ensure output length matches input length (padding might cause mismatch)
        if len(denoised_intensity) != n_points:
             logging.warning(f"Wavelet reconstruction length ({len(denoised_intensity)}) differs from input ({n_points}). Adjusting length.")
             if len(denoised_intensity) > n_points:
                  denoised_intensity = denoised_intensity[:n_points] # Truncate
             else: # Pad with edge value if shorter (slightly better than zero)
                  pad_width = n_points - len(denoised_intensity)
                  denoised_intensity = np.pad(denoised_intensity, (0, pad_width), mode='edge')

        # Restore original NaNs/Infs in the output
        denoised_intensity[~finite_mask] = intensity[~finite_mask]

        logging.info("Wavelet denoising applied successfully.")
        return denoised_intensity

    except ImportError: # Should be caught by PYWAVELETS_AVAILABLE check
        logging.error("PyWavelets not available during denoise execution.")
        return intensity
    except Exception as e:
        logging.error(f"Error during wavelet denoising: {e}", exc_info=True)
        return intensity # Return original data on failure


# --- Noise Analysis ---

def analyze_noise(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    signal_free_regions: Optional[List[Tuple[float, float]]] = None
) -> Tuple[Optional[float], List[Tuple[float, float]]]:
    """
    Analyzes noise level (standard deviation) in specified signal-free regions.

    Args:
        wavelengths (np.ndarray): Wavelength array.
        intensity (np.ndarray): Intensity array (typically raw or baseline-corrected).
        signal_free_regions (Optional[List[Tuple[float, float]]]):
            List of tuples, each defining a region (start_wl, end_wl) assumed
            to contain only noise.

    Returns:
        Tuple[Optional[float], List[Tuple[float, float]]]:
            - Estimated noise standard deviation across all valid regions, or None if failed.
            - List of the valid regions actually used for the calculation.
    """
    if not isinstance(wavelengths, np.ndarray) or not isinstance(intensity, np.ndarray):
        raise TypeError("Wavelengths and intensity must be NumPy arrays.")
    if wavelengths.shape != intensity.shape:
        raise ValueError("Wavelength and Intensity arrays must have the same shape.")

    if signal_free_regions is None or not signal_free_regions:
        logging.warning("No signal-free regions provided for noise analysis. Cannot estimate noise.")
        return None, [] # Return empty list for regions used

    noise_segments = []
    valid_regions_used: List[Tuple[float, float]] = []

    for i, region in enumerate(signal_free_regions):
        try:
            # Validate region format
            if not isinstance(region, (tuple, list)) or len(region) != 2:
                 logging.warning(f"Skipping invalid noise region format at index {i}: {region}. Expected (start_wl, end_wl).")
                 continue
            start_wl, end_wl = map(float, region) # Attempt conversion

            # Validate region values
            if not (np.isfinite(start_wl) and np.isfinite(end_wl)):
                 logging.warning(f"Skipping invalid noise region {region} at index {i}: Non-finite limits.")
                 continue
            if start_wl >= end_wl:
                 logging.warning(f"Skipping invalid noise region {region} at index {i}: start >= end.")
                 continue

            # Extract data within region
            region_mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
            intensity_in_region = intensity[region_mask]

            if intensity_in_region.size == 0:
                 logging.debug(f"Skipping noise region {region} at index {i}: No data points found in this wavelength range.")
                 continue

            # Consider only finite points within the region for std dev calculation
            finite_intensity_in_region = intensity_in_region[np.isfinite(intensity_in_region)]
            if finite_intensity_in_region.size < 2: # Need at least 2 points for std dev
                 logging.warning(f"Skipping noise region {region} at index {i}: Fewer than 2 finite data points for std dev calculation.")
                 continue

            # Store the finite segment and mark region as used
            noise_segments.append(finite_intensity_in_region)
            valid_regions_used.append(region) # Store original valid region tuple

        except (TypeError, ValueError) as e:
            logging.warning(f"Skipping invalid noise region {region} at index {i}: Error parsing limits ({e}).")
            continue
        except Exception as e:
            logging.error(f"Unexpected error processing noise region {region} at index {i}: {e}", exc_info=True)
            continue

    # Calculate overall std dev if any valid segments found
    if not noise_segments:
        logging.error("Could not extract any valid data points from the specified signal-free regions.")
        return None, []

    try:
        all_noise_points = np.concatenate(noise_segments)
        if all_noise_points.size < 2: # Double check after concatenation
            logging.error("Insufficient total finite data points (< 2) across all valid noise regions.")
            return None, valid_regions_used

        # Calculate standard deviation using numpy
        noise_std_dev = np.std(all_noise_points, ddof=1) # Use ddof=1 for sample standard deviation? Or 0 for population? Let's use 1.

        if not np.isfinite(noise_std_dev):
             logging.error(f"Noise standard deviation calculation resulted in non-finite value: {noise_std_dev}.")
             return None, valid_regions_used

        logging.info(f"Estimated noise standard deviation: {noise_std_dev:.4f} "
                     f"(from {len(valid_regions_used)} regions, {all_noise_points.size} points).")
        return float(noise_std_dev), valid_regions_used

    except Exception as e:
        logging.error(f"Failed to calculate final noise standard deviation: {e}", exc_info=True)
        return None, valid_regions_used

# --- Example Usage (Optional) ---
# Removed for brevity, can be added back if needed for testing.