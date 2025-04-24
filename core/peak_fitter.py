# -*- coding: utf-8 -*-
"""
Peak fitting functions for individual peaks in LIBS spectra.

Includes profile definitions (Gaussian, Lorentzian, Pseudo-Voigt), fitting logic
using scipy.optimize.curve_fit, ROI determination, local baseline correction,
and model selection based on AIC/BIC.
"""

import logging
import numpy as np
import pandas as pd # Not used in this file, keep? Remove for cleanliness.
import traceback
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
# Removed dataclass import as FitResult is now in data_models
from enum import Enum

# --- SciPy Imports ---
SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit
    from scipy.special import voigt_profile # Keep if true Voigt might be added later
    from scipy.integrate import trapezoid # Use trapezoid for numerical integration (area)
    # Alternative: from scipy.integrate import quad # More precise but needs function
    SCIPY_AVAILABLE = True
except ImportError:
    logging.error("SciPy library not found. Peak fitting functionality unavailable. Install with 'pip install scipy'.")
    # Provide placeholder functions that raise ImportError if called
    def curve_fit(*args, **kwargs): raise ImportError("SciPy required but not installed.")
    def voigt_profile(*args, **kwargs): raise ImportError("SciPy required but not installed.")
    def trapezoid(*args, **kwargs): raise ImportError("SciPy required but not installed.")

# --- Local Imports ---
try:
    # Use relative imports assuming standard structure
    from .data_models import Spectrum, FitResult # Import FitResult from data_models
    # Import ProfileType Enum and other constants/helpers if defined elsewhere
except ImportError as e_import:
    logging.critical(f"CRITICAL ERROR in peak_fitter.py: Cannot import core dependencies: {e_import}.")
    raise ImportError(f"Core dependencies failed to import in peak_fitter: {e_import}") from e_import

# --- Constants and Configuration ---
FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ~2.35482
# Moved MIN_ROI_POINTS to config.yaml, use a default here if needed as fallback
DEFAULT_MIN_ROI_POINTS = 5
MIN_WIDTH_NM = 1e-6      # Minimum allowed width parameter during fitting to avoid numerical issues
EPSILON = 1e-9           # Small value for numerical stability checks
DEFAULT_MAX_ITER = 2000  # Default max iterations for curve_fit
LOCAL_BASELINE_EDGE_POINTS = 3 # Number of points at each edge for LocalLinear baseline
LOCAL_BASELINE_MAX_FRAC = 0.25 # Max fraction of ROI points used by edges for LocalLinear

# --- Enums for Configuration (Mirroring data_models if not imported) ---
# If Enums are defined centrally (e.g., in data_models or a dedicated file), import them instead.
class ProfileType(Enum):
    GAUSSIAN = "Gaussian"
    LORENTZIAN = "Lorentzian"
    PSEUDO_VOIGT = "PseudoVoigt"
    # VOIGT = "Voigt" # Add if implementing true Voigt

class BaselineMode(Enum):
    NONE = "None"
    LOCAL_LINEAR = "LocalLinear" # Fit linear baseline to edges of ROI
    SLOPE = "Slope"          # Simple linear baseline between ROI endpoints

class ModelSelectionCriterion(Enum):
    AIC = "AIC"
    BIC = "BIC"


# --- Profile Functions (Gaussian, Lorentzian, Pseudo-Voigt) ---
# These functions define the shapes used for fitting.

def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """Gaussian profile function (amplitude, center, standard deviation sigma)."""
    # Bounds in curve_fit should prevent sigma <= 0
    # Add explicit check for safety, though redundant if bounds are correct
    if sigma <= EPSILON: return np.full_like(x, np.inf) # Or 0? Inf signals issue better.
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))

def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    """Lorentzian profile function (amplitude, center, HWHM gamma)."""
    # Bounds in curve_fit should prevent gamma <= 0
    if gamma <= EPSILON: return np.full_like(x, np.inf)
    return amplitude * (gamma**2 / ((x - center)**2 + gamma**2))

def pseudo_voigt(x: np.ndarray, amplitude: float, center: float, sigma: float, eta: float) -> np.ndarray:
    """
    Pseudo-Voigt profile using Gaussian sigma and a mixing parameter eta (0=Gauss, 1=Lorentz).
    FWHM is derived from sigma and eta (using approximation in FitResult).

    Args:
        x: Wavelengths.
        amplitude: Peak amplitude.
        center: Peak center wavelength.
        sigma: Standard deviation of the Gaussian component.
        eta: Mixing parameter (0 <= eta <= 1).

    Returns:
        Calculated Pseudo-Voigt profile values.
    """
    # Bounds will handle sigma > 0 and 0 <= eta <= 1
    if sigma <= EPSILON: return np.full_like(x, np.inf)
    eta_bounded = np.clip(eta, 0.0, 1.0) # Ensure eta is valid

    # Calculate gamma (Lorentzian HWHM) consistently based on Gaussian FWHM
    # FWHM_G = sigma * FWHM_GAUSS_FACTOR
    # FWHM_L = gamma * 2
    # Setting FWHM_L approx FWHM_G => gamma approx sigma * FWHM_GAUSS_FACTOR / 2
    gamma = max(EPSILON, sigma * FWHM_GAUSS_FACTOR / 2.0)

    gauss_part = (1.0 - eta_bounded) * np.exp(-((x - center)**2) / (2 * sigma**2))
    loren_part = eta_bounded * (gamma**2 / ((x - center)**2 + gamma**2))
    return amplitude * (gauss_part + loren_part)

# Dictionary mapping ProfileType Enum to function and expected parameter count
# Ensures consistency and simplifies dispatching.
PROFILE_FUNCTIONS: Dict[ProfileType, Tuple[Callable, int]] = {
    ProfileType.GAUSSIAN: (gaussian, 3),
    ProfileType.LORENTZIAN: (lorentzian, 3),
    ProfileType.PSEUDO_VOIGT: (pseudo_voigt, 4),
}

# --- Goodness-of-Fit Calculations ---

def calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R-squared (coefficient of determination).

    Handles edge cases like constant data or insufficient points.

    Args:
        y_true: Array of true data values.
        y_pred: Array of predicted values from the model.

    Returns:
        R-squared value (float between 0 and 1), or NaN if input is invalid.
    """
    # Input validation
    if y_true is None or y_pred is None or y_true.shape != y_pred.shape or len(y_true) < 2:
        return np.nan

    # Check for NaNs/Infs - should be filtered before calling this ideally
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.all(valid_mask):
         logging.warning("R² calculation received non-finite values. Calculating on finite subset.")
         y_true = y_true[valid_mask]
         y_pred = y_pred[valid_mask]
         if len(y_true) < 2: return np.nan # Not enough valid points

    # Calculate R-squared
    ss_res = np.sum((y_true - y_pred)**2)
    mean_y_true = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y_true)**2)

    # Handle edge case: constant data (ss_tot is zero)
    if abs(ss_tot) < EPSILON:
        return 1.0 if abs(ss_res) < EPSILON else 0.0

    r2 = 1.0 - (ss_res / ss_tot)
    return float(max(0.0, r2)) # Ensure R2 is not negative due to numerical issues

def calculate_aic_bic(n: int, k: int, rss: float) -> Tuple[float, float]:
    """
    Calculates AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)
    using the residual sum of squares (RSS). Assumes errors are normally distributed.

    Handles perfect fits (RSS ~ 0) by returning -Inf.

    Args:
        n (int): Number of data points used for fitting.
        k (int): Number of fitted parameters in the model.
        rss (float): Residual Sum of Squares from the fit.

    Returns:
        Tuple[float, float]: Calculated AIC and BIC values. Returns (Inf, Inf) if
                             input parameters are invalid (e.g., n <= k). Returns
                             (-Inf, -Inf) for perfect fits (RSS ~ 0).
    """
    if n <= k or n == 0 or not np.isfinite(rss): # Check n > k for validity
        # Cannot compute if points <= parameters, or RSS is invalid
        logging.debug(f"AIC/BIC not computable (n={n}, k={k}, rss={rss}). Returning Inf.")
        return np.inf, np.inf

    # Check for perfect fit (RSS effectively zero)
    if rss < EPSILON:
        logging.debug(f"AIC/BIC: RSS is near zero ({rss:.2e}). Indicating perfect fit (-Inf).")
        return -np.inf, -np.inf # Perfect fit should always win model selection

    try:
        # Standard formula based on likelihood for normally distributed errors
        # Using estimated variance = rss / n
        log_likelihood_term = n * np.log(rss / n) # Can be negative if rss/n < 1
        aic = 2 * k + log_likelihood_term
        bic = k * np.log(n) + log_likelihood_term

        # Optional: AICc (Corrected AIC for small sample sizes)
        # if (n - k - 1) > 0:
        #     aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
        # else:
        #     aicc = np.inf # AICc not defined
        # Consider returning aicc as well if needed

        if not (np.isfinite(aic) and np.isfinite(bic)):
             logging.warning(f"AIC/BIC calculation resulted in non-finite values (AIC={aic}, BIC={bic}). Returning Inf.")
             return np.inf, np.inf
        return float(aic), float(bic)

    except (ValueError, FloatingPointError) as e: # Catch potential math errors (e.g., log of non-positive)
        logging.warning(f"AIC/BIC calculation failed (n={n}, k={k}, rss={rss}): {e}", exc_info=True)
        return np.inf, np.inf


# --- Helper Functions for Fitting ---

def _determine_roi(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    peak_index: int,
    roi_factor: float,
    min_roi_width_nm: float,
    min_roi_points: int, # Get from config now
    explicit_roi_wl: Optional[Tuple[float, float]] = None
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """
    Determines the Region of Interest (ROI) indices and wavelengths around a peak.

    Handles explicit ROI definition or automatic calculation based on estimated FWHM.
    Performs edge-aware expansion if the initial ROI is too small.

    Args:
        wavelengths: Full wavelength array.
        intensity: Full intensity array (e.g., processed intensity).
        peak_index: Index of the peak maximum.
        roi_factor: Multiplier for estimated FWHM for auto ROI width.
        min_roi_width_nm: Minimum width (nm) for auto ROI.
        min_roi_points: Minimum number of data points required in the final ROI.
        explicit_roi_wl: Optional tuple (min_wl, max_wl) to force ROI.

    Returns:
        Tuple containing (roi_indices, wl_roi, int_roi, center_wl_guess) or None if ROI is invalid.
    """
    center_wl_guess = wavelengths[peak_index]
    n_points = len(wavelengths)

    if explicit_roi_wl and len(explicit_roi_wl) == 2 and explicit_roi_wl[0] < explicit_roi_wl[1]:
        # --- Use explicitly provided ROI wavelengths ---
        roi_start_wl, roi_end_wl = explicit_roi_wl
        # Find indices within the explicit range
        roi_indices = np.where((wavelengths >= roi_start_wl) & (wavelengths <= roi_end_wl))[0]
        logging.debug(f"Peak @ {center_wl_guess:.3f}nm: Using provided ROI [{roi_start_wl:.3f}-{roi_end_wl:.3f}]nm")
        if len(roi_indices) == 0:
             logging.error(f"Provided ROI [{roi_start_wl:.3f}-{roi_end_wl:.3f}]nm contains no data points. Cannot fit.")
             return None
    else:
        # --- Automatic ROI calculation based on estimated FWHM ---
        try:
            # Estimate amplitude and baseline near the peak for FWHM calc
            # Look slightly wider to better estimate local baseline/minimum
            local_range_pts = 15 # Look +/- 15 points
            local_min_idx = max(0, peak_index - local_range_pts)
            local_max_idx = min(n_points, peak_index + local_range_pts + 1) # Exclusive end
            local_intensity = intensity[local_min_idx:local_max_idx]
            local_wavelengths = wavelengths[local_min_idx:local_max_idx]

            # Handle all-NaN local range
            finite_local_mask = np.isfinite(local_intensity)
            if not np.any(finite_local_mask):
                logging.warning(f"Peak @ {center_wl_guess:.3f}nm: Cannot estimate amplitude/FWHM near peak due to all NaNs in local range [{local_wavelengths[0]:.2f}-{local_wavelengths[-1]:.2f}]nm.")
                # Fallback: use a fixed number of points around the peak index
                roi_half_points = 10 # Arbitrary fallback width in points
                fwhm_guess_nm = abs(wavelengths[min(n_points-1, peak_index + roi_half_points)] - wavelengths[max(0, peak_index - roi_half_points)])
            else:
                local_intensity_finite = local_intensity[finite_local_mask]
                local_min_intensity = np.min(local_intensity_finite) # Min of finite points
                peak_intensity_val = intensity[peak_index]

                # Ensure peak intensity is finite and use it, otherwise fallback
                if not np.isfinite(peak_intensity_val):
                     logging.warning(f"Peak intensity at index {peak_index} is non-finite. Using max in local range for amplitude guess.")
                     peak_intensity_val = np.max(local_intensity_finite) # Fallback to local max
                     # Re-find peak_index if needed? Assumes original index is still okay.

                # Amplitude guess relative to local minimum
                amplitude_guess = peak_intensity_val - local_min_intensity
                amplitude_guess = max(amplitude_guess, EPSILON) # Ensure positive

                half_max_val = local_min_intensity + amplitude_guess / 2.0

                # Find indices where intensity crosses half max using interpolation
                # Ensure we use finite data for interpolation base
                local_finite_wl = local_wavelengths[finite_local_mask]
                local_finite_int = local_intensity_finite

                # Interpolate intensity values at fine steps around the peak to find crossing points robustly
                interp_wl = np.linspace(local_finite_wl[0], local_finite_wl[-1], len(local_finite_wl) * 5) # Finer grid
                interp_int = np.interp(interp_wl, local_finite_wl, local_finite_int)

                # Find where interpolated intensity crosses half_max_val
                above_half_indices = np.where(interp_int >= half_max_val)[0]
                if len(above_half_indices) > 1:
                    wl_left_cross = interp_wl[above_half_indices[0]]
                    wl_right_cross = interp_wl[above_half_indices[-1]]
                    fwhm_guess_nm = abs(wl_right_cross - wl_left_cross)
                else: # Fallback if crossing not found well
                     logging.warning(f"Could not reliably find half-max crossings for peak @ {center_wl_guess:.3f}nm. Using fallback width.")
                     fwhm_guess_nm = abs(local_wavelengths[-1] - local_wavelengths[0]) / 3.0 # Guess ~1/3 of local range width

            # Ensure FWHM guess is reasonable (at least pixel width)
            min_pixel_width = np.min(np.diff(wavelengths)) if n_points > 1 else MIN_WIDTH_NM
            fwhm_guess_nm = max(fwhm_guess_nm, min_pixel_width, MIN_WIDTH_NM) # Ensure positive non-zero

            # Calculate ROI bounds based on FWHM guess
            roi_half_width_nm = max((roi_factor / 2.0) * fwhm_guess_nm, min_roi_width_nm / 2.0)
            roi_start_wl = center_wl_guess - roi_half_width_nm
            roi_end_wl = center_wl_guess + roi_half_width_nm
            roi_indices = np.where((wavelengths >= roi_start_wl) & (wavelengths <= roi_end_wl))[0]

        except Exception as e:
            logging.error(f"Error calculating auto ROI for peak @ {center_wl_guess:.3f}nm: {e}", exc_info=True)
            return None

    # --- Validate and potentially expand ROI size ---
    if len(roi_indices) < min_roi_points:
        logging.warning(f"Peak @ {center_wl_guess:.3f}nm: Initial ROI too narrow ({len(roi_indices)} pts, need {min_roi_points}). Attempting edge-aware expansion.")
        needed = min_roi_points - len(roi_indices)

        start_idx = roi_indices[0] if len(roi_indices) > 0 else peak_index
        end_idx = roi_indices[-1] if len(roi_indices) > 0 else peak_index

        # Expand intelligently, respecting boundaries
        current_left = start_idx
        current_right = end_idx
        for _ in range(needed):
            # Calculate distance to edges
            dist_left = current_left
            dist_right = (n_points - 1) - current_right
            # Prefer expanding away from closer edge, or expand both if centered
            if dist_left < dist_right and current_right < n_points - 1:
                 current_right += 1
            elif dist_right < dist_left and current_left > 0:
                 current_left -= 1
            # If equidistant or only one direction possible, expand where possible
            elif current_left > 0:
                 current_left -= 1
            elif current_right < n_points - 1:
                 current_right += 1
            else:
                 break # Cannot expand further in either direction

        final_start_idx = current_left
        final_end_idx = current_right
        roi_indices = np.arange(final_start_idx, final_end_idx + 1)

        if len(roi_indices) < min_roi_points:
            logging.error(f"Peak @ {center_wl_guess:.3f}nm: ROI still too narrow ({len(roi_indices)} pts) after edge-aware expansion. Cannot fit.")
            return None

    wl_roi = wavelengths[roi_indices]
    int_roi = intensity[roi_indices] # Use original intensity within ROI for baseline correction

    # --- Final check for sufficient *finite* data points in ROI ---
    num_finite_in_roi = np.sum(np.isfinite(int_roi))
    if num_finite_in_roi < min_roi_points:
        logging.error(f"Peak @ {center_wl_guess:.3f}nm: Not enough finite data points ({num_finite_in_roi}) in the final ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm. Need {min_roi_points}. Cannot fit.")
        return None

    logging.debug(f"Peak @ {center_wl_guess:.3f}nm: Final ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm ({len(wl_roi)} pts, {num_finite_in_roi} finite).")
    return roi_indices, wl_roi, int_roi, center_wl_guess

def _apply_local_baseline(
    wl_roi: np.ndarray,
    int_roi: np.ndarray,
    baseline_mode: BaselineMode
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies local baseline correction within the ROI.

    Args:
        wl_roi: Wavelengths within the Region of Interest.
        int_roi: Corresponding intensity values within the ROI.
        baseline_mode: The baseline method to apply locally.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Intensity ROI corrected for local baseline.
            - Calculated local baseline array.
    """
    local_baseline = np.zeros_like(int_roi) # Default baseline is zero
    int_roi_corrected = int_roi.copy() # Start with original intensity

    n_roi_points = len(wl_roi)
    mode_failed = False # Flag to track if chosen mode fails

    if baseline_mode == BaselineMode.LOCAL_LINEAR:
        # Use a fixed number of points at edges, minimum 2, max fraction of ROI
        num_edge = max(2, LOCAL_BASELINE_EDGE_POINTS)
        max_edge_points = int(n_roi_points * LOCAL_BASELINE_MAX_FRAC)
        num_edge = min(num_edge, max_edge_points // 2) # Ensure total edge points <= max_frac

        if n_roi_points >= max(4, 2 * num_edge): # Need enough points overall and for edges
            try:
                # Define edge indices carefully
                edge_idx = np.concatenate((np.arange(num_edge), np.arange(n_roi_points - num_edge, n_roi_points)))
                edge_wl, edge_int = wl_roi[edge_idx], int_roi[edge_idx]

                # Filter out NaNs before polyfit
                finite_mask = np.isfinite(edge_wl) & np.isfinite(edge_int)
                if np.sum(finite_mask) >= 2: # Need at least 2 finite points for linear fit
                    coeffs = np.polyfit(edge_wl[finite_mask], edge_int[finite_mask], 1)
                    local_baseline = np.polyval(coeffs, wl_roi)
                else:
                     logging.warning(f"LocalLinear baseline failed: Not enough finite edge points ({np.sum(finite_mask)}) in ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm.")
                     mode_failed = True
            except (np.linalg.LinAlgError, ValueError) as e:
                logging.warning(f"LocalLinear baseline polyfit failed for ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm: {e}. Check edge data.")
                mode_failed = True
            except Exception as e:
                 logging.error(f"Unexpected error during LocalLinear baseline: {e}", exc_info=True)
                 mode_failed = True
        else:
             logging.warning(f"LocalLinear baseline failed: ROI too small ({n_roi_points} pts) for edge calculation (need >= {max(4, 2 * num_edge)} pts, num_edge={num_edge}).")
             mode_failed = True

    elif baseline_mode == BaselineMode.SLOPE:
        if n_roi_points >= 2:
            wl_diff = wl_roi[-1] - wl_roi[0]
            int_start, int_end = int_roi[0], int_roi[-1]
            # Check if endpoints and difference are valid
            if abs(wl_diff) > EPSILON and np.isfinite(int_start) and np.isfinite(int_end):
                slope = (int_end - int_start) / wl_diff
                local_baseline = int_start + slope * (wl_roi - wl_roi[0])
            else:
                 logging.warning(f"Slope baseline failed (zero width={wl_diff:.2e} or NaN endpoints=[{int_start}, {int_end}]) for ROI [{wl_roi[0]:.3f}-{wl_roi[-1]:.3f}]nm.")
                 mode_failed = True
        else: # Need at least 2 points for slope
             logging.warning(f"Slope baseline failed: ROI too small ({n_roi_points} pts).")
             mode_failed = True

    # Handle failure or NONE mode
    if mode_failed or baseline_mode == BaselineMode.NONE:
        if mode_failed: logging.warning("Falling back to no local baseline correction.")
        local_baseline = np.zeros_like(int_roi) # Ensure baseline is zero
        int_roi_corrected = int_roi # Ensure corrected is same as original ROI intensity
    else:
        # Apply correction only if baseline calculation succeeded
        int_roi_corrected = int_roi - local_baseline

    # Ensure corrected intensity doesn't contain NaNs introduced by baseline calc?
    # If int_roi was finite, polyval should produce finite baseline.
    # Replace potential NaNs in corrected with original values if needed.
    nan_mask_corrected = ~np.isfinite(int_roi_corrected)
    if np.any(nan_mask_corrected):
         logging.warning("NaNs found in locally baseline-corrected ROI intensity. Replacing with original values at those points.")
         int_roi_corrected[nan_mask_corrected] = int_roi[nan_mask_corrected]

    return int_roi_corrected, local_baseline

def _get_initial_guesses_and_bounds(
    profile_type: ProfileType,
    wl_fit: np.ndarray,         # Use filtered wavelengths for guesses/bounds
    int_fit: np.ndarray,         # Use filtered intensity for guesses/bounds
    # Removed center_wl_guess as input, calculate from filtered data
) -> Optional[Tuple[List[float], Tuple[List[float], List[float]]]]:
    """
    Estimates initial parameters (p0) and bounds for curve_fit based on filtered ROI data.

    Args:
        profile_type: The profile function type to generate guesses/bounds for.
        wl_fit: Filtered wavelength array within ROI (no NaNs/Infs).
        int_fit: Filtered, baseline-corrected intensity array within ROI (no NaNs/Infs).

    Returns:
        Optional Tuple containing:
        - p0 (List[float]): Initial parameter guesses.
        - bounds (Tuple[List[float], List[float]]): Lower and upper parameter bounds.
        Returns None if guesses or bounds cannot be reasonably determined.
    """
    n_fit_points = len(wl_fit)
    if n_fit_points < PROFILE_FUNCTIONS[profile_type][1]: # Need enough points for parameters
         logging.error(f"Cannot get guesses: Not enough finite points ({n_fit_points}) in fit ROI for {profile_type.value}.")
         return None

    # --- Calculate Guesses from Filtered Data ---
    try:
        # Center Guess: Wavelength at max intensity within filtered ROI
        max_idx = np.argmax(int_fit)
        center_wl_guess = wl_fit[max_idx]

        # Amplitude Guess: Max intensity value in filtered ROI
        amp_guess = int_fit[max_idx] # Already baseline-corrected
        amp_guess = max(amp_guess, EPSILON) # Ensure positive

        # Width Guess (FWHM based on filtered data)
        half_max_val = np.min(int_fit) + (amp_guess - np.min(int_fit)) / 2.0
        above_half_max_indices = np.where(int_fit >= half_max_val)[0]

        if len(above_half_max_indices) > 1:
            fwhm_guess_nm = abs(wl_fit[above_half_max_indices[-1]] - wl_fit[above_half_max_indices[0]])
        else: # Fallback width if FWHM hard to estimate
            fwhm_guess_nm = abs(wl_fit[-1] - wl_fit[0]) / 3.0 # ~1/3 of fit data range

        # Ensure minimum width based on data resolution or absolute floor
        min_pixel_width = np.min(np.diff(wl_fit)) if n_fit_points > 1 else MIN_WIDTH_NM
        fwhm_guess_nm = max(fwhm_guess_nm, min_pixel_width, MIN_WIDTH_NM)

        # Guesses for sigma (Gauss/PV) and gamma (Lorentz)
        sigma_guess = fwhm_guess_nm / FWHM_GAUSS_FACTOR
        gamma_guess = fwhm_guess_nm / 2.0
        eta_guess = 0.5 # Default mixing for PV

        # Check if guesses are finite
        if not all(np.isfinite([center_wl_guess, amp_guess, sigma_guess, gamma_guess, eta_guess])):
             logging.error(f"Non-finite initial parameter guess calculated for {profile_type.value}. Aborting guess generation.")
             return None

    except Exception as e:
        logging.error(f"Error calculating initial guesses for {profile_type.value}: {e}", exc_info=True)
        return None

    # --- Define Bounds ---
    min_wl_roi, max_wl_roi = wl_fit[0], wl_fit[-1]
    roi_width_nm = max_wl_roi - min_wl_roi

    # Width Bounds
    min_width_bound = MIN_WIDTH_NM
    # Max width bound: slightly less than full ROI width to avoid edge cases
    max_width_bound = max(roi_width_nm * 0.95, min_width_bound * 2)

    # Amplitude Bounds
    min_amp_bound = 0.0 # Cannot be negative
    # Max amplitude: allow significantly larger than initial guess
    max_amp_bound = amp_guess * 10.0

    # Center Bounds: Tighten around the guess, but allow movement within ROI
    center_buffer = roi_width_nm * 0.1 # Allow +/- 10% of ROI width around guess
    center_min_bound = max(min_wl_roi, center_wl_guess - center_buffer)
    center_max_bound = min(max_wl_roi, center_wl_guess + center_buffer)
    # Ensure bounds are valid
    if center_min_bound >= center_max_bound: center_min_bound, center_max_bound = min_wl_roi, max_wl_roi

    # Ensure all bounds are finite
    bounds_list = [min_amp_bound, max_amp_bound, center_min_bound, center_max_bound, min_width_bound, max_width_bound]
    if not all(np.isfinite(bounds_list)):
         logging.error(f"Non-finite parameter bound calculated for {profile_type.value}. Aborting guess generation. Bounds: {bounds_list}")
         return None


    # --- Clip Guesses & Assemble p0/bounds ---
    sigma_guess = np.clip(sigma_guess, min_width_bound, max_width_bound)
    gamma_guess = np.clip(gamma_guess, min_width_bound, max_width_bound)
    amp_guess = np.clip(amp_guess, min_amp_bound, max_amp_bound)
    center_wl_guess = np.clip(center_wl_guess, center_min_bound, center_max_bound)

    p0: List[float]
    bounds: Tuple[List[float], List[float]]

    if profile_type == ProfileType.GAUSSIAN:
        p0 = [amp_guess, center_wl_guess, sigma_guess]
        bounds = ([min_amp_bound, center_min_bound, min_width_bound],
                  [max_amp_bound, center_max_bound, max_width_bound])
    elif profile_type == ProfileType.LORENTZIAN:
        p0 = [amp_guess, center_wl_guess, gamma_guess]
        bounds = ([min_amp_bound, center_min_bound, min_width_bound],
                  [max_amp_bound, center_max_bound, max_width_bound])
    elif profile_type == ProfileType.PSEUDO_VOIGT:
        p0 = [amp_guess, center_wl_guess, sigma_guess, eta_guess]
        bounds = ([min_amp_bound, center_min_bound, min_width_bound, 0.0], # Eta bounds [0, 1]
                  [max_amp_bound, center_max_bound, max_width_bound, 1.0])
    else:
        logging.error(f"Unsupported profile type for guessing: {profile_type}")
        return None

    # Final check: ensure p0 has finite values
    if not all(np.isfinite(p0)):
         logging.error(f"Non-finite value in final p0 for {profile_type.value}: {p0}. Aborting guess generation.")
         return None

    logging.debug(f"Guesses for {profile_type.value}: p0={p0}, bounds=({bounds[0]}, {bounds[1]})")
    return p0, bounds


def _perform_single_fit(
    profile_type: ProfileType,
    wl_roi: np.ndarray,         # Original ROI wavelengths (for context/guess recalc)
    int_roi_corrected: np.ndarray, # Baseline-corrected intensity in ROI
    max_iterations: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Performs curve_fit for a single profile type on the provided ROI data.

    Filters NaNs, gets initial guesses based on filtered data, runs curve_fit,
    and validates the resulting covariance matrix.

    Args:
        profile_type: The profile to fit.
        wl_roi: Original wavelengths in the ROI.
        int_roi_corrected: Baseline-corrected intensity in the ROI.
        max_iterations: Max iterations for the fitter.

    Returns:
        Tuple containing:
        - Optimized parameters (np.ndarray) or None on failure.
        - Covariance matrix (np.ndarray) or None on failure/invalid result.
        - Status message (str).
    """
    fit_func, n_params = PROFILE_FUNCTIONS[profile_type]

    # --- Filter out NaN values before fitting - curve_fit cannot handle them ---
    finite_mask = np.isfinite(wl_roi) & np.isfinite(int_roi_corrected)
    n_finite_points = np.sum(finite_mask)

    if n_finite_points < n_params: # Need at least as many points as parameters
         return None, None, f"Fit failed: Not enough finite data points ({n_finite_points}) in ROI for {profile_type.value} fit (need {n_params})."

    wl_fit = wl_roi[finite_mask]
    int_fit = int_roi_corrected[finite_mask]

    # --- Get Guesses/Bounds based *only* on the filtered data ---
    guess_result = _get_initial_guesses_and_bounds(profile_type, wl_fit, int_fit)
    if guess_result is None:
         return None, None, f"Fit failed: Could not determine valid initial guesses/bounds for {profile_type.value}."
    p0, bounds = guess_result

    # --- Perform Fit ---
    try:
        params_opt, params_cov = curve_fit(
            fit_func,
            wl_fit,  # Use filtered data
            int_fit,   # Use filtered data
            p0=p0,
            bounds=bounds,
            maxfev=max_iterations,
            method='trf', # Trust Region Reflective handles bounds well
            # sigma=weights, # Optional: Provide weights based on data uncertainty
            # check_finite=True, # Default True, ensures inputs are finite
        )

        # --- Validate Results ---
        # Check optimized parameters are finite
        if not np.all(np.isfinite(params_opt)):
             return None, None, f"Fit converged but produced non-finite parameters for {profile_type.value}."

        # Check covariance matrix validity (exists, is finite, has positive diagonal)
        # Note: If curve_fit fails to estimate covariance, it returns inf diagonal.
        if params_cov is None or not np.all(np.isfinite(params_cov)) or np.any(np.diag(params_cov) <= 0):
             logging.warning(f"Fit converged for {profile_type.value}, but covariance matrix is invalid "
                             f"(e.g., non-finite or non-positive variance). Treating as failed fit quality.")
             # Return parameters, but None for covariance to signal the issue
             return params_opt, None, f"{profile_type.value} fit covariance invalid."

        return params_opt, params_cov, f"{profile_type.value} fit successful."

    except (RuntimeError, ValueError) as e:
        # RuntimeError often means optimal parameters not found (max iterations, bad jacobian)
        # ValueError can arise from input shape mismatches or issues within the model function
        msg = f"Fit optimization failed for {profile_type.value}: {e}"
        logging.debug(msg) # Log details at debug level
        return None, None, msg
    except Exception as e:
        # Catch any other unexpected errors during fitting
        msg = f"Unexpected error during {profile_type.value} fit: {e}"
        logging.error(msg, exc_info=True)
        return None, None, msg

# --- Area Calculation Helper ---
def _calculate_fit_area(fit_result: FitResult) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculates the area (integral) of the fitted profile.

    Uses numerical integration (trapezoid rule) over a fine grid within
    a reasonable range around the peak center.
    Area error estimation is currently basic/placeholder.

    Args:
        fit_result: The successful FitResult object.

    Returns:
        Tuple[Optional[float], Optional[float]]: Estimated area and area error (or None).
    """
    if not fit_result.success or not all(np.isfinite([fit_result.amplitude, fit_result.center, fit_result.width])):
        return None, None

    try:
        profile_func = PROFILE_FUNCTIONS[fit_result.profile_type][0]
        params = [fit_result.amplitude, fit_result.center, fit_result.width]
        if fit_result.profile_type == ProfileType.PSEUDO_VOIGT:
            if fit_result.mixing_param_eta is None or not np.isfinite(fit_result.mixing_param_eta): return None, None
            params.append(np.clip(fit_result.mixing_param_eta, 0.0, 1.0))

        # Define integration range: +/- several widths around the center
        # Use FWHM if available and valid, otherwise estimate from width param
        fwhm = fit_result.fwhm if fit_result.fwhm is not None and np.isfinite(fit_result.fwhm) and fit_result.fwhm > 0 else None
        if fwhm is None: # Estimate FWHM roughly if needed
             if fit_result.profile_type == 'Gaussian': fwhm = fit_result.width * FWHM_GAUSS_FACTOR
             elif fit_result.profile_type == 'Lorentzian': fwhm = fit_result.width * 2.0
             elif fit_result.profile_type == 'PseudoVoigt': fwhm = fit_result.width * FWHM_GAUSS_FACTOR # Approximation
             else: fwhm = fit_result.width * 3 # Generic guess if width param meaning is unknown

        half_width_for_integration = max(fwhm * 4.0, fit_result.width * 8.0) # Integrate over +/- 4 FWHM or +/- 8 sigma/gamma
        num_integration_points = 500 # Number of points for numerical integration

        x_integrate = np.linspace(
            fit_result.center - half_width_for_integration,
            fit_result.center + half_width_for_integration,
            num_integration_points
        )
        y_integrate = profile_func(x_integrate, *params)

        # Use trapezoid rule for numerical integration
        area = trapezoid(y_integrate, x_integrate)

        # --- Area Error Estimation (Basic Placeholder) ---
        # Proper error propagation requires Jacobian of the integral w.r.t parameters and covariance matrix.
        # Simple approximation: Assume area error is proportional to amplitude error.
        area_err = None
        if fit_result.param_errors and len(fit_result.param_errors) > 0 and np.isfinite(fit_result.param_errors[0]):
             amp_err = fit_result.param_errors[0]
             # Relative error of area approx = relative error of amplitude
             if np.isfinite(fit_result.amplitude) and abs(fit_result.amplitude) > EPSILON:
                 area_err = abs(area * (amp_err / fit_result.amplitude))
             else:
                  area_err = np.nan # Cannot estimate relative error if amplitude is zero/invalid
        # --- End Placeholder ---

        logging.debug(f"Calculated area for {fit_result.profile_type} @ {fit_result.center:.2f}nm: {area:.3f} ± {area_err if area_err is not None else 'N/A'}")
        return float(area) if np.isfinite(area) else None, float(area_err) if area_err is not None and np.isfinite(area_err) else None

    except Exception as e:
        logging.error(f"Error calculating area for fit {fit_result}: {e}", exc_info=True)
        return None, None

def _evaluate_fit(
    profile_type: ProfileType,
    params: np.ndarray,
    cov: Optional[np.ndarray], # Covariance might be None if quality check failed
    wl_roi: np.ndarray,         # Original ROI wl (for context and evaluation)
    int_roi_corrected: np.ndarray # Original ROI corrected intensity (for metrics)
) -> FitResult:
    """Calculates metrics, area, and creates a FitResult object for a successful fit."""

    fit_func, n_params = PROFILE_FUNCTIONS[profile_type]

    # Calculate fitted curve on the original ROI wavelengths (including any NaNs)
    # Use only finite points from int_roi_corrected for calculation, put NaN elsewhere
    y_fitted = np.full_like(wl_roi, np.nan)
    finite_mask = np.isfinite(wl_roi) & np.isfinite(int_roi_corrected)
    if np.any(finite_mask):
        try:
             # Calculate Y values only where input intensity was finite
             y_fitted[finite_mask] = fit_func(wl_roi[finite_mask], *params)
        except Exception as e:
             logging.error(f"Error calculating y_fitted for {profile_type.value}: {e}", exc_info=True)
             # y_fitted will remain NaN

    # Calculate metrics only on the finite points used for fitting
    int_fit = int_roi_corrected[finite_mask]
    y_fit_points = y_fitted[finite_mask] # Get corresponding fitted points

    # Check if y_fit_points calculation succeeded before calculating metrics
    if not np.all(np.isfinite(y_fit_points)):
         logging.warning(f"Fit evaluation metrics skipped for {profile_type.value}: Non-finite values in fitted curve.")
         rss = np.nan
         r2 = np.nan
         aic = np.inf
         bic = np.inf
    else:
         rss = np.sum((int_fit - y_fit_points)**2)
         r2 = calculate_r_squared(int_fit, y_fit_points)
         n_fit_points = len(int_fit)
         aic, bic = calculate_aic_bic(n_fit_points, n_params, rss)

    # Extract common parameters
    amplitude = params[0]
    center = params[1]
    width = params[2] # Sigma for Gauss/PV, Gamma for Lorentz
    eta = params[3] if profile_type == ProfileType.PSEUDO_VOIGT else None

    # Create FitResult instance
    fit_result_obj = FitResult(
        profile_type=profile_type.value, # Store string value
        amplitude=float(amplitude),
        center=float(center),
        width=float(width),
        mixing_param_eta=float(eta) if eta is not None else None,
        params_covariance=cov, # Store cov matrix (even if None/invalid, post_init checks it)
        r_squared=float(r2),
        aic=float(aic),
        bic=float(bic),
        success=True, # Marked as success because fitter converged
        message=f"{profile_type.value} fit evaluation complete.",
        # Store context
        roi_wavelengths=wl_roi,
        roi_intensity_corrected=int_roi_corrected,
        fitted_curve=y_fitted,
        # Area and Area Error are initially None, calculated below
        area=None,
        area_err=None,
    )

    # --- Calculate Area ---
    # Requires the initial FitResult object for parameters
    peak_area, peak_area_err = _calculate_fit_area(fit_result_obj)
    fit_result_obj.area = peak_area
    fit_result_obj.area_err = peak_area_err
    # --- End Area Calculation ---

    # __post_init__ in FitResult will calculate fwhm and param_errors
    return fit_result_obj


# --- Main Fitting Function ---

def fit_peak(
    spectrum: Spectrum,
    peak_index: int,
    processed_intensity: np.ndarray,
    roi_factor: float = 7.0,
    min_roi_width_nm: float = 0.1,
    min_roi_points: int = DEFAULT_MIN_ROI_POINTS, # Get from config
    profiles_to_fit: Optional[List[Union[str, ProfileType]]] = None, # Allow string or enum
    model_selection_criterion: Union[str, ModelSelectionCriterion] = ModelSelectionCriterion.AIC, # Allow string or enum
    baseline_mode: Union[str, BaselineMode] = BaselineMode.LOCAL_LINEAR, # Allow string or enum
    max_fit_iterations: int = DEFAULT_MAX_ITER,
    roi_wavelengths: Optional[Tuple[float, float]] = None
) -> Tuple[Optional[FitResult], Dict[str, FitResult]]:
    """
    Fits specified profiles to a single peak within a defined ROI of the processed intensity.

    Performs local baseline correction within the ROI before fitting.
    Selects the best model based on the specified criterion (AIC or BIC).

    Args:
        spectrum: Spectrum object containing wavelengths.
        peak_index: Index of the estimated peak center in the spectrum arrays.
        processed_intensity: Intensity array (e.g., baseline-corrected) to fit.
        roi_factor: Multiplier for estimated FWHM to determine auto ROI width. Ignored if roi_wavelengths is set.
        min_roi_width_nm: Minimum width of the auto ROI in nm. Ignored if roi_wavelengths is set.
        min_roi_points: Minimum number of data points required in the ROI.
        profiles_to_fit: Profiles to attempt fitting (list of strings or ProfileType enums).
                         Defaults to all implemented types if None.
        model_selection_criterion: Criterion ('AIC', 'BIC', or enum) for best model selection.
        baseline_mode: Method ('LocalLinear', 'Slope', 'None', or enum) for local baseline subtraction.
        max_fit_iterations: Max iterations for curve_fit.
        roi_wavelengths: Explicit ROI [min_wl, max_wl]. Overrides auto ROI if provided.

    Returns:
        Tuple[Optional[FitResult], Dict[str, FitResult]]:
            - Best FitResult object based on the selection criterion (or None if all fits fail).
            - Dictionary {profile_type_str: FitResult} for all attempted fits (including failures).
    """
    if not SCIPY_AVAILABLE:
        logging.error("Cannot fit peak: SciPy library is unavailable.")
        return None, {}

    # --- Validate Inputs ---
    if not isinstance(spectrum, Spectrum) or spectrum.wavelengths is None or processed_intensity is None:
         logging.error("Invalid input: Spectrum or processed_intensity is missing.")
         return None, {}
    if not (0 <= peak_index < len(spectrum.wavelengths)) or not (0 <= peak_index < len(processed_intensity)):
        logging.error(f"Invalid peak index: {peak_index} for data length {len(spectrum.wavelengths)}")
        return None, {}
    if len(spectrum.wavelengths) != len(processed_intensity):
        logging.error(f"Shape mismatch: Wavelengths ({len(spectrum.wavelengths)}) vs Intensity ({len(processed_intensity)})")
        return None, {}

    # --- Resolve Enums/Strings ---
    try:
        # Ensure baseline_mode is the Enum member
        if isinstance(baseline_mode, str): baseline_mode = BaselineMode(baseline_mode)
        # Ensure model_selection_criterion is the Enum member
        if isinstance(model_selection_criterion, str): model_selection_criterion = ModelSelectionCriterion(model_selection_criterion)
        # Standardize profiles_to_fit to list of ProfileType enums
        if profiles_to_fit is None:
            profiles_enum_list = list(PROFILE_FUNCTIONS.keys())
        else:
            profiles_enum_list = []
            for p in profiles_to_fit:
                if isinstance(p, ProfileType): profiles_enum_list.append(p)
                elif isinstance(p, str): profiles_enum_list.append(ProfileType(p)) # Convert string to enum
                else: logging.warning(f"Ignoring invalid profile type in list: {p}")
            if not profiles_enum_list:
                 logging.error("No valid profiles specified in profiles_to_fit list.")
                 return None, {}
    except ValueError as e_enum:
         logging.error(f"Invalid parameter string value: {e_enum}. Check config options.")
         return None, {} # Fail if parameters are invalid

    # --- 1. Determine Region of Interest (ROI) ---
    try:
        roi_result = _determine_roi(
            wavelengths=spectrum.wavelengths,
            intensity=processed_intensity, # Use the provided processed intensity for ROI determination
            peak_index=peak_index,
            roi_factor=roi_factor,
            min_roi_width_nm=min_roi_width_nm,
            min_roi_points=min_roi_points,
            explicit_roi_wl=roi_wavelengths
        )
        if roi_result is None:
            # Error already logged in _determine_roi
            logging.error(f"Failed to determine valid ROI for peak index {peak_index}.")
            return None, {} # Return empty dict for all_results if ROI fails
        roi_indices, wl_roi, int_roi_raw, center_wl_guess = roi_result

    except Exception as e_roi:
        logging.error(f"Unexpected error defining ROI for peak index {peak_index}: {e_roi}", exc_info=True)
        return None, {}

    # --- 2. Local Baseline Correction ---
    try:
        int_roi_corrected, _ = _apply_local_baseline(wl_roi, int_roi_raw, baseline_mode)
    except Exception as e_base:
         logging.error(f"Unexpected error applying local baseline for peak index {peak_index}: {e_base}", exc_info=True)
         # Proceed without local correction? Or fail? Let's proceed but log clearly.
         logging.warning("Proceeding without local baseline correction due to error.")
         int_roi_corrected = int_roi_raw # Use the uncorrected ROI intensity


    # --- 3. Fit Specified Profiles ---
    all_results: Dict[str, FitResult] = {} # Keyed by profile type string

    for profile_type_enum in profiles_enum_list:
        profile_type_str = profile_type_enum.value # Use string key for dict

        if profile_type_enum not in PROFILE_FUNCTIONS:
            logging.warning(f"Skipping unsupported profile type enum: {profile_type_enum}")
            all_results[profile_type_str] = FitResult(profile_type=profile_type_str, amplitude=np.nan, center=np.nan, width=np.nan, success=False, message="Unsupported profile type")
            continue

        logging.debug(f"Attempting {profile_type_str} fit for peak near {center_wl_guess:.3f}nm...")
        params_opt, params_cov, fit_message = _perform_single_fit(
            profile_type=profile_type_enum,
            wl_roi=wl_roi,
            int_roi_corrected=int_roi_corrected,
            max_iterations=max_fit_iterations
        )

        if params_opt is not None:
            # Fit converged, now evaluate and store result
            try:
                result = _evaluate_fit(
                    profile_type=profile_type_enum,
                    params=params_opt,
                    cov=params_cov, # Pass cov (might be None if quality check failed)
                    wl_roi=wl_roi,
                    int_roi_corrected=int_roi_corrected
                )
                # Ensure result is marked success=False if cov was invalid
                result.success = (params_cov is not None) # Success requires valid covariance
                if not result.success: result.message = fit_message # Keep message if cov invalid
                all_results[profile_type_str] = result
                logging.debug(f"Evaluated {profile_type_str}. Success={result.success}. Message: {result.message}")
            except Exception as e_eval:
                logging.error(f"Error evaluating successful {profile_type_str} fit: {e_eval}", exc_info=True)
                all_results[profile_type_str] = FitResult(profile_type=profile_type_str, amplitude=params_opt[0], center=params_opt[1], width=params_opt[2], success=False, message=f"Evaluation error: {e_eval}")
        else:
            # Fit failed during optimization
            logging.warning(f"Fit failed for {profile_type_str} near {center_wl_guess:.3f}nm: {fit_message}")
            all_results[profile_type_str] = FitResult(profile_type=profile_type_str, amplitude=np.nan, center=np.nan, width=np.nan, success=False, message=fit_message)


    # --- 4. Select Best Fit ---
    best_fit_result: Optional[FitResult] = None
    best_score = np.inf

    # Consider only fits marked as successful (includes covariance check)
    successful_fits = [res for res in all_results.values() if res.success]

    if not successful_fits:
        logging.warning(f"No successful fits (with valid covariance) found for peak near {center_wl_guess:.3f}nm.")
        # Return None for best_fit, but include all failed attempts in all_results
        return None, all_results

    for result in successful_fits:
        score = np.inf
        # Use the selected criterion enum
        if model_selection_criterion == ModelSelectionCriterion.AIC:
            score = result.aic if result.aic is not None else np.inf
        elif model_selection_criterion == ModelSelectionCriterion.BIC:
            score = result.bic if result.bic is not None else np.inf

        # Check if score is finite and better than current best
        # Handle -np.inf score for perfect fits correctly (it should win)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_fit_result = result
        elif score == -np.inf and best_score != -np.inf: # Perfect fit automatically wins
            best_score = score
            best_fit_result = result

    if best_fit_result:
        score_str = f"{best_score:.2f}" if np.isfinite(best_score) else str(best_score)
        logging.info(f"Best fit selected for peak near {center_wl_guess:.3f}nm: {best_fit_result.profile_type} "
                     f"({model_selection_criterion.value}={score_str}, R2={best_fit_result.r_squared:.3f})")
    else:
        # This might happen if all successful fits had non-finite AIC/BIC scores
        logging.warning(f"Could not determine best fit among successful fits for peak near {center_wl_guess:.3f}nm (likely Inf scores). Returning None.")

    # Return the best fit (or None) and the dictionary containing all attempted results
    return best_fit_result, all_results

# Note: Example Usage block removed for brevity, can be added back if needed for standalone testing.
# Note: lmfit integration suggestion:
# Consider adding lmfit as an optional backend for more complex fitting scenarios.
# This would involve:
# - Defining lmfit Models (e.g., GaussianModel(), LorentzianModel(), PseudoVoigtModel()).
# - Setting parameter hints (initial values, bounds) using `model.make_params()`.
# - Running `model.fit(data, params, x=x_data)`.
# - Extracting results (parameters, errors, fit statistics) from the lmfit ModelResult object.
# This could improve robustness and flexibility, especially for overlapping peaks or constrained fits.