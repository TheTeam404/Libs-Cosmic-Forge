# -*- coding: utf-8 -*-
"""
Core signal processing functions for LIBS spectra.

Includes baseline correction, smoothing, peak profile functions,
and noise analysis capabilities.
"""

import logging
import numpy as np
from typing import Tuple, Optional, List, Union

# --- SciPy Import Handling ---
# Check for SciPy availability, needed for Savitzky-Golay smoothing.
try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning(
        "SciPy not found. Savitzky-Golay smoothing will be unavailable. "
        "Install with 'pip install scipy'."
    )
    # Define a placeholder function to raise an error if called
    def savgol_filter(*args, **kwargs):
        raise ImportError(
            "SciPy is required for Savitzky-Golay smoothing but is not installed."
        )

# --- Utility Import ---
# Assuming 'utils.helpers.ensure_odd' exists and ensures an integer is odd.
# If not available, replace with: def ensure_odd(n): return n if n % 2 != 0 else n + 1
try:
    from utils.helpers import ensure_odd
except ImportError:
    logging.warning("utils.helpers.ensure_odd not found. Using basic implementation.")
    def ensure_odd(n: int) -> int:
        """Ensures an integer is odd."""
        n = int(n)
        return n if n % 2 != 0 else n + 1

# --- Constants ---
# Factor to convert Gaussian sigma to FWHM
FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0)) # Approx 2.35482
# Small epsilon to avoid division by zero or log(0) issues
EPSILON = 1e-9

# --- Profile Functions ---
# These functions define standard peak shapes used in spectral analysis.

def gaussian(x: np.ndarray, amplitude: float, center: float, sigma: float) -> np.ndarray:
    """
    Calculates a Gaussian profile.

    Args:
        x: Array of independent variable values (e.g., wavelength, Raman shift).
        amplitude: Peak height at the center.
        center: Position of the peak maximum.
        sigma: Standard deviation of the Gaussian distribution.

    Returns:
        Calculated Gaussian profile values corresponding to x.
        Returns np.inf where sigma is non-positive.
    """
    if sigma <= EPSILON:
        # Return infinity or a very large number where profile is undefined
        # np.inf helps in optimization routines to avoid these regions
        logging.warning(f"Gaussian sigma ({sigma:.2e}) too small or non-positive. Returning Inf.")
        return np.full_like(x, np.inf)
    return amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))

def lorentzian(x: np.ndarray, amplitude: float, center: float, gamma: float) -> np.ndarray:
    """
    Calculates a Lorentzian profile.

    Args:
        x: Array of independent variable values.
        amplitude: Peak height at the center.
        center: Position of the peak maximum.
        gamma: Half-width at half-maximum (HWHM).

    Returns:
        Calculated Lorentzian profile values corresponding to x.
        Returns np.inf where gamma is non-positive.
    """
    if gamma <= EPSILON:
        logging.warning(f"Lorentzian gamma ({gamma:.2e}) too small or non-positive. Returning Inf.")
        return np.full_like(x, np.inf)
    return amplitude * (gamma**2 / ((x - center)**2 + gamma**2))

def pseudo_voigt(
    x: np.ndarray, amplitude: float, center: float, sigma: float, eta: float
) -> np.ndarray:
    """
    Calculates a Pseudo-Voigt profile, a linear combination of Gaussian and Lorentzian.

    Args:
        x: Array of independent variable values.
        amplitude: Peak height at the center.
        center: Position of the peak maximum.
        sigma: Standard deviation of the Gaussian component.
        eta: Mixing coefficient (0 <= eta <= 1).
             eta = 0 gives a pure Gaussian.
             eta = 1 gives a pure Lorentzian.

    Returns:
        Calculated Pseudo-Voigt profile values corresponding to x.
        Returns np.inf where sigma is non-positive.
    """
    if sigma <= EPSILON:
        logging.warning(f"Pseudo-Voigt sigma ({sigma:.2e}) too small or non-positive. Returning Inf.")
        return np.full_like(x, np.inf)

    eta_bounded = np.clip(eta, 0.0, 1.0) # Ensure eta is within [0, 1]

    # Derive the corresponding Lorentzian HWHM (gamma) from Gaussian sigma
    # This ensures comparable widths when eta changes
    gamma = max(EPSILON, (sigma * FWHM_GAUSS_FACTOR) / 2.0)

    gauss_part = (1.0 - eta_bounded) * np.exp(-((x - center)**2) / (2 * sigma**2))
    loren_part = eta_bounded * (gamma**2 / ((x - center)**2 + gamma**2))

    return amplitude * (gauss_part + loren_part)


# --- Baseline Correction Algorithms ---

def _interpolate_finite(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Internal helper to interpolate NaN/Inf values using linear interpolation."""
    y_interp = y.copy()
    finite_mask = np.isfinite(y_interp)
    all_finite = np.all(finite_mask)

    if not all_finite:
        nan_mask = ~finite_mask
        try:
            # Get indices of finite and non-finite points
            finite_indices = np.flatnonzero(finite_mask)
            nan_indices = np.flatnonzero(nan_mask)

            if len(finite_indices) < 2:
                # Cannot interpolate with fewer than 2 finite points
                logging.error("Cannot interpolate NaNs: Fewer than 2 finite data points.")
                # Replace NaNs with 0 as a fallback, might not be ideal
                y_interp[nan_mask] = 0.0
                # Recompute finite mask if needed, although it might be all False now
                finite_mask = np.isfinite(y_interp)
                if not np.any(finite_mask): # Check if everything became non-finite
                   logging.error("Interpolation fallback failed, all data seems non-finite.")
                   return y, np.zeros_like(y, dtype=bool), False # Return original and failure flag

            else:
                 # Perform linear interpolation
                y_interp[nan_mask] = np.interp(nan_indices, finite_indices, y_interp[finite_indices])
                logging.warning(f"Interpolated {np.sum(nan_mask)} NaN/Inf values using linear interpolation.")

        except Exception as e:
            logging.error(f"Error during NaN interpolation: {e}. Returning data with NaNs potentially zeroed.", exc_info=True)
            # Attempt to zero out remaining NaNs as a last resort
            y_interp[~np.isfinite(y_interp)] = 0.0
            finite_mask = np.isfinite(y_interp) # Recompute mask
            if not np.any(finite_mask):
                return y, np.zeros_like(y, dtype=bool), False # Return original if all else fails

        # Verify if interpolation succeeded
        if not np.all(np.isfinite(y_interp)):
            logging.error("Failed to remove all NaNs/Infs through interpolation and fallback. Errors may occur.")
            # Proceed with potentially remaining NaNs zeroed out
            finite_mask = np.isfinite(y_interp) # Update mask based on current state

    return y_interp, finite_mask, True # Return interpolated, original finite mask, success flag


def baseline_poly(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    order: int = 3,
    percentile: float = 10.0,
    max_iterations: int = 10, # Added for iterative refinement (optional)
    tolerance: float = 0.001, # Added for iterative refinement (optional)
    **kwargs # Accept unused kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates baseline using polynomial fitting on points below a percentile.

    This method identifies points below a specified percentile of the intensity
    and fits a polynomial to these points. It can optionally iterate to
    refine the baseline estimate.

    Args:
        wavelengths: Wavelength array.
        intensity: Intensity array.
        order: Order of the polynomial to fit (e.g., 1 for linear, 3 for cubic).
               If order < 0, baseline correction is skipped.
        percentile: Percentile threshold (0-100) to select baseline points.
        max_iterations: (Optional) Maximum iterations for refinement. If > 1,
                       the process iteratively refits the baseline to points
                       below the *previous* baseline estimate plus a small margin.
        tolerance: (Optional) Tolerance for convergence if using iterations.
        kwargs: Accepts extra arguments for compatibility.

    Returns:
        Tuple (corrected_intensity, baseline_estimate). Returns (original_intensity,
        zero_baseline) if correction is skipped or fails.
    """
    if order < 0:
        logging.debug("Polynomial baseline skipped (order < 0).")
        return intensity, np.zeros_like(intensity)

    if not isinstance(wavelengths, np.ndarray) or not isinstance(intensity, np.ndarray):
        raise TypeError("Wavelengths and intensity must be NumPy arrays.")

    if wavelengths.shape != intensity.shape:
        raise ValueError(
            f"Wavelength ({wavelengths.shape}) and Intensity ({intensity.shape}) "
            "arrays must have the same shape."
        )

    n_points = len(wavelengths)
    min_points_required = order + 1

    if n_points < min_points_required:
        logging.warning(
            f"Polynomial baseline skipped: Need at least {min_points_required} points "
            f"for order {order}, but got {n_points}."
        )
        return intensity, np.zeros_like(intensity)

    if not (0 < percentile <= 100):
        original_percentile = percentile
        percentile = 10.0
        logging.warning(
            f"Invalid percentile ({original_percentile}). Using default: {percentile}%."
        )

    # Work on copies and handle non-finite values
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
        logging.error("Failed to handle NaNs/Infs. Skipping polynomial baseline.")
        return intensity, np.zeros_like(intensity)

    # --- Iterative Baseline Refinement (Optional) ---
    # Based on pybaselines implementation concept
    current_intensity = intensity_processed
    baseline = np.zeros_like(intensity_processed) # Initial baseline guess

    for iteration in range(max_iterations if max_iterations > 0 else 1):
        try:
            # Use only originally finite points for percentile calculation initially
            # In later iterations, this might use the current corrected signal
            valid_for_percentile = intensity_processed[original_finite_mask] if iteration == 0 else current_intensity[original_finite_mask]
            if len(valid_for_percentile) == 0:
                 logging.error("No finite points available to calculate percentile. Skipping baseline.")
                 return intensity, np.zeros_like(intensity)

            threshold = np.percentile(valid_for_percentile, percentile)

            # Select points below the threshold for fitting
            # Use the *currently processed* intensity for mask selection
            mask = (current_intensity <= threshold) & original_finite_mask # Always respect original NaNs
            n_masked_pts = np.sum(mask)

            # --- Fallback if too few points are selected ---
            if n_masked_pts < min_points_required:
                logging.warning(
                    f"Iteration {iteration+1}: Only {n_masked_pts} points below {percentile:.1f}% "
                    f"percentile (threshold={threshold:.2f}). Need {min_points_required}. "
                    "Using lowest intensity points as fallback."
                )
                # Find indices of the lowest intensity points among the originally finite ones
                finite_indices = np.where(original_finite_mask)[0]
                if len(finite_indices) < min_points_required:
                    logging.error(
                        f"Fallback failed: Only {len(finite_indices)} finite points available. "
                        "Cannot fit polynomial. Skipping baseline."
                    )
                    return intensity, np.zeros_like(intensity)

                # Sort finite points by intensity and take the lowest ones
                sorted_finite_indices = finite_indices[np.argsort(intensity_processed[finite_indices])]
                fallback_indices = sorted_finite_indices[:min_points_required]

                # Create a new mask based on these fallback indices
                mask = np.zeros_like(intensity_processed, dtype=bool)
                mask[fallback_indices] = True
                n_masked_pts = np.sum(mask) # Should be min_points_required
                logging.debug(f"Fallback selected {n_masked_pts} lowest finite points.")


            # Extract points for fitting
            x_masked = wavelengths[mask]
            y_masked = intensity_processed[mask] # Use the imputed intensity for fitting

            # Final check for safety before polyfit
            if not (np.all(np.isfinite(x_masked)) and np.all(np.isfinite(y_masked))):
                 logging.error("Non-finite values detected in data selected for polyfit after masking/fallback. Skipping.")
                 return intensity, np.zeros_like(intensity)
            if len(x_masked) < min_points_required: # Should be caught above, but double check
                logging.error("Insufficient points for polyfit after final checks. Skipping.")
                return intensity, np.zeros_like(intensity)


            # --- Polynomial Fitting ---
            coeffs = np.polyfit(x_masked, y_masked, order)
            new_baseline = np.polyval(coeffs, wavelengths)

            # --- Check for Convergence (if iterating) ---
            if max_iterations > 1:
                diff = np.abs(baseline - new_baseline)
                # Check convergence on finite points only
                if np.all(diff[original_finite_mask] < tolerance):
                    logging.info(f"Polynomial baseline converged after {iteration + 1} iterations.")
                    baseline = new_baseline
                    break
            # Update for next iteration or final result
            baseline = new_baseline
            # Update the signal being considered for the next iteration's percentile
            current_intensity = intensity_processed - baseline

        except (np.linalg.LinAlgError, ValueError) as e:
            logging.error(f"Polynomial baseline fitting failed: {e}", exc_info=True)
            return intensity, np.zeros_like(intensity)
        except Exception as e:
            logging.error(f"Unexpected error during polynomial baseline: {e}", exc_info=True)
            return intensity, np.zeros_like(intensity)

    # --- Final Steps ---
    # Ensure baseline is not above the *processed* intensity (important for peaks)
    # This step might be debated, but often desired. Consider if it should be optional.
    # baseline = np.minimum(baseline, intensity_processed) # Optional constraint

    corrected_intensity = intensity_processed - baseline

    # Restore original NaNs/Infs in the output
    corrected_intensity[~original_finite_mask] = intensity[~original_finite_mask]
    baseline[~original_finite_mask] = np.nan # Baseline is undefined where original was NaN

    logging.info(
        f"Polynomial baseline applied (order={order}, percentile={percentile:.1f}%, "
        f"iterations={iteration+1 if max_iterations > 0 else 1})."
    )
    return corrected_intensity, baseline


def baseline_snip(
    wavelengths: np.ndarray, # Kept for API consistency, but not used by SNIP
    intensity: np.ndarray,
    max_iterations: int = 100,
    increasing_window: bool = True,
    **kwargs # Accept unused kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates baseline using the SNIP algorithm.

    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) iteratively
    smooths the spectrum, effectively removing peaks to estimate the underlying
    baseline.

    Based on the algorithm described by Ryan et al., Computers & Chemistry 12 (1988)
    and variations (e.g., Morháč et al., NIM A 471 (2001)).

    Args:
        wavelengths: Wavelength array (unused by SNIP, kept for consistency).
        intensity: Intensity array.
        max_iterations: Number of clipping iterations (window sizes) to apply.
                        Controls the degree of smoothing / peak removal.
        increasing_window: If True (default), the clipping window size increases
                           from 1 to `max_iterations`. If False, a fixed window
                           size of `max_iterations` is used in each iteration (less common).
        kwargs: Accepts extra arguments for compatibility.

    Returns:
        Tuple (baseline_corrected_intensity, calculated_baseline).
        Returns (original_intensity, zero_baseline) if correction fails.
    """
    if not isinstance(intensity, np.ndarray):
        raise TypeError("Intensity must be a NumPy array.")

    n_points = len(intensity)
    if n_points < 3:
        logging.warning(f"SNIP requires >= 3 points, got {n_points}. Skipping.")
        return intensity, np.zeros_like(intensity)

    if max_iterations < 1:
         logging.warning(f"SNIP max_iterations ({max_iterations}) must be >= 1. Setting to 1.")
         max_iterations = 1

    # Work on a copy and handle non-finite values by interpolation
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
        logging.error("Failed to handle NaNs/Infs. Skipping SNIP baseline.")
        return intensity, np.zeros_like(intensity)

    baseline = intensity_processed.copy() # Start baseline estimate from processed signal

    logging.info(
        f"Starting SNIP baseline estimation (max_iter={max_iterations}, "
        f"increasing_window={increasing_window})."
    )

    try:
        # --- SNIP Iterations ---
        for k in range(1, max_iterations + 1):
            window_size = k if increasing_window else max_iterations

            # Stop if window size becomes too large (e.g., half the spectrum)
            if window_size >= n_points // 2:
                logging.debug(
                    f"SNIP iteration {k}: Window size ({window_size}) reached limit. Stopping."
                )
                break

            prev_baseline = baseline.copy() # Store previous iteration's baseline

            # Apply the core SNIP clipping filter operation efficiently using slicing
            # For each point y[i], compare it with the average of y[i-k] and y[i+k]
            # Vectorized equivalent of the inner loop:
            avg_neighbors = (prev_baseline[:-2 * window_size] + prev_baseline[2 * window_size:]) / 2.0
            points_to_clip = prev_baseline[window_size:-window_size] > avg_neighbors
            baseline[window_size:-window_size][points_to_clip] = avg_neighbors[points_to_clip]


            # --- Optional Convergence Check ---
            # If needed, uncomment and potentially add a tolerance parameter
            # if np.allclose(baseline, prev_baseline, atol=1e-6):
            #     logging.info(f"SNIP converged after {k} iterations.")
            #     break

        # --- Final Steps ---
        # Ensure the final baseline does not exceed the processed intensity
        baseline = np.minimum(baseline, intensity_processed)

        corrected_intensity = intensity_processed - baseline

        # Restore original NaNs/Infs in the output
        corrected_intensity[~original_finite_mask] = intensity[~original_finite_mask]
        baseline[~original_finite_mask] = np.nan # Baseline undefined where original was NaN

        logging.info("SNIP baseline estimation complete.")
        return corrected_intensity, baseline

    except Exception as e:
        logging.error(f"Error during SNIP baseline correction: {e}", exc_info=True)
        # Return original intensity and zero baseline on failure
        return intensity, np.zeros_like(intensity)


# --- Smoothing Algorithms ---

def smooth_savitzky_golay(
    intensity: np.ndarray,
    smoothing_window: int = 11,
    smoothing_polyorder: int = 3,
    **kwargs # Accept unused kwargs
) -> np.ndarray:
    """
    Applies Savitzky-Golay smoothing filter to the intensity data.

    Handles NaNs/Infs by linear interpolation before applying the filter.

    Args:
        intensity: Intensity array to smooth.
        smoothing_window: The length of the filter window (must be a positive odd integer).
                          Will be adjusted if even, too small, or too large.
        smoothing_polyorder: The order of the polynomial used to fit the samples
                             (must be less than window_length). Will be adjusted if needed.
        kwargs: Accepts extra arguments for compatibility.

    Returns:
        The smoothed intensity array. Returns the original array if smoothing
        cannot be performed (e.g., SciPy unavailable, data too short, errors).
    """
    if not SCIPY_AVAILABLE:
        logging.error("Cannot smooth: SciPy library is unavailable.")
        return intensity # Return original data if SciPy isn't installed

    if not isinstance(intensity, np.ndarray):
        raise TypeError("Intensity must be a NumPy array.")

    n_points = len(intensity)
    if n_points < 3:
        logging.warning(f"Data too short ({n_points} points) for Sav-Gol smoothing. Returning original.")
        return intensity

    # --- Parameter Validation and Adjustment ---
    try:
        # Ensure window is an odd integer >= 3
        wl = ensure_odd(int(smoothing_window))
        if wl < 3:
            logging.warning(f"Sav-Gol window ({smoothing_window}) too small. Adjusting to 3.")
            wl = 3

        # Ensure polyorder is an integer >= 0
        po = int(smoothing_polyorder)
        if po < 0:
             logging.warning(f"Sav-Gol polyorder ({smoothing_polyorder}) cannot be negative. Adjusting to 0 (moving average).")
             po = 0 # A Sav-Gol filter with order 0 is equivalent to a moving average

        # Ensure window is not larger than data length
        if wl > n_points:
            old_wl = wl
            # Adjust window to largest possible odd integer <= n_points
            wl = n_points if n_points % 2 != 0 else n_points - 1
            wl = max(3, wl) # Ensure it's still at least 3
            logging.warning(
                f"Sav-Gol window ({old_wl}) > data length ({n_points}). "
                f"Adjusting window to {wl}."
            )

        # Ensure polyorder is less than window length
        if po >= wl:
            old_po = po
            # Adjust polyorder to be wl - 1 or lower (use wl - 2 for more smoothing bias)
            po = max(0, wl - 2) # Adjust down, ensure non-negative
            logging.warning(
                f"Sav-Gol polyorder ({old_po}) >= adjusted window ({wl}). "
                f"Adjusting polyorder to {po}."
            )

    except (ValueError, TypeError) as e:
        logging.error(f"Invalid Sav-Gol parameters (window={smoothing_window}, order={smoothing_polyorder}): {e}. Returning original.")
        return intensity

    # --- NaN/Inf Handling ---
    intensity_processed, original_finite_mask, interp_ok = _interpolate_finite(intensity)
    if not interp_ok:
         logging.error("Failed to handle NaNs/Infs prior to smoothing. Returning original.")
         return intensity
    elif not np.all(original_finite_mask):
         logging.debug("Applied linear interpolation to handle NaNs/Infs before smoothing.")


    # --- Apply Savitzky-Golay Filter ---
    try:
        # Check again if adjusted parameters are valid for the potentially reduced finite data length
        # Although interpolation should handle this, it's a safeguard.
        if wl > len(intensity_processed) or po >= wl:
             logging.error(f"Internal Error: Adjusted Sav-Gol parameters (wl={wl}, po={po}) still invalid after processing. Skipping.")
             # This case should ideally not be reached if validation and interpolation work correctly
             return intensity # Return original non-smoothed intensity


        smoothed_intensity = savgol_filter(intensity_processed, window_length=wl, polyorder=po)

        # Restore original NaNs/Infs in the output
        smoothed_intensity[~original_finite_mask] = intensity[~original_finite_mask]

        logging.info(f"Applied Savitzky-Golay smoothing (window={wl}, order={po}).")
        return smoothed_intensity

    except ValueError as e:
        # Catches errors from savgol_filter itself (e.g., if somehow params are still invalid)
        logging.error(f"Error applying Savitzky-Golay filter: {e}. Returning original.", exc_info=True)
        return intensity
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"Unexpected error during Savitzky-Golay smoothing: {e}", exc_info=True)
        return intensity


# --- Noise Analysis ---

def analyze_noise(
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    signal_free_regions: Optional[List[Tuple[float, float]]] = None
) -> Tuple[Optional[float], Optional[List[Tuple[float, float]]]]:
    """
    Estimates noise level from signal-free regions of the spectrum.

    Calculates the standard deviation of the intensity within specified
    wavelength regions assumed to contain only noise (and baseline).

    Args:
        wavelengths: Wavelength array corresponding to the intensity.
        intensity: Intensity array (ideally baseline-corrected, but not required).
        signal_free_regions: A list of tuples, where each tuple defines the
                             start and end wavelength of a signal-free region.
                             Example: [(200.0, 210.0), (450.0, 455.5)]

    Returns:
        Tuple containing:
        - Estimated noise standard deviation (float) or None if calculation fails.
        - The list of valid signal-free regions used (or None).
        Returns (None, signal_free_regions) if no valid regions are provided or found.
    """
    if not isinstance(wavelengths, np.ndarray) or not isinstance(intensity, np.ndarray):
        raise TypeError("Wavelengths and intensity must be NumPy arrays.")
    if wavelengths.shape != intensity.shape:
        raise ValueError("Wavelength and Intensity arrays must have the same shape.")
    if signal_free_regions is None or not signal_free_regions:
        logging.warning("No signal-free regions provided for noise analysis. Cannot estimate noise.")
        return None, signal_free_regions

    noise_segments = []
    valid_regions_used = []

    for region in signal_free_regions:
        try:
            start_wl, end_wl = map(float, region)
            if start_wl >= end_wl:
                logging.warning(f"Skipping invalid noise region {region}: start >= end.")
                continue

            # Find indices corresponding to the wavelength region
            region_mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
            intensity_in_region = intensity[region_mask]

            if intensity_in_region.size == 0:
                logging.warning(f"Skipping noise region {region}: No data points found in this range.")
                continue

            # Exclude NaNs/Infs within the region before calculating std dev
            finite_intensity_in_region = intensity_in_region[np.isfinite(intensity_in_region)]

            if finite_intensity_in_region.size < 2:
                 logging.warning(f"Skipping noise region {region}: Fewer than 2 finite data points for std dev calculation.")
                 continue

            noise_segments.append(finite_intensity_in_region)
            valid_regions_used.append(region)

        except (TypeError, ValueError) as e:
            logging.warning(f"Skipping invalid noise region {region}: Error parsing limits ({e}).")
            continue
        except Exception as e:
             logging.error(f"Unexpected error processing noise region {region}: {e}", exc_info=True)
             continue


    if not noise_segments:
        logging.error("Could not find any valid data points in the specified signal-free regions.")
        return None, valid_regions_used # Return the list of regions that were attempted

    # Concatenate all valid noise segments and calculate overall standard deviation
    try:
        all_noise_points = np.concatenate(noise_segments)
        if all_noise_points.size < 2:
             logging.error("Insufficient total finite data points (< 2) across all valid noise regions.")
             return None, valid_regions_used

        noise_std_dev = np.std(all_noise_points)
        logging.info(
            f"Estimated noise standard deviation: {noise_std_dev:.4f} "
            f"(from {len(valid_regions_used)} regions, {all_noise_points.size} points)."
        )
        return noise_std_dev, valid_regions_used

    except Exception as e:
        logging.error(f"Failed to calculate final noise standard deviation: {e}", exc_info=True)
        return None, valid_regions_used


# --- Example Usage (Optional) ---
if __name__ == '__main__':
    # Configure logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

    # Generate sample data (Sine wave + Gaussian peak + Noise + Baseline)
    np.random.seed(42)
    wavelengths = np.linspace(200, 800, 1000)
    baseline_true = 50 + 0.1 * (wavelengths - 200) # Linear baseline
    signal = (
        gaussian(wavelengths, amplitude=200, center=350, sigma=5) +
        gaussian(wavelengths, amplitude=150, center=600, sigma=10) +
        lorentzian(wavelengths, amplitude=80, center=700, gamma=8)
    )
    noise = np.random.normal(0, 5, size=wavelengths.shape) # Noise std dev = 5
    intensity_raw = signal + baseline_true + noise

    # --- Test Baseline Correction ---
    print("\n--- Testing Polynomial Baseline ---")
    intensity_poly_corrected, baseline_poly_calc = baseline_poly(
        wavelengths, intensity_raw, order=2, percentile=5.0, max_iterations=1 # Single iteration test
    )

    print("\n--- Testing SNIP Baseline ---")
    intensity_snip_corrected, baseline_snip_calc = baseline_snip(
        wavelengths, intensity_raw, max_iterations=50, increasing_window=True
    )

    # --- Test Smoothing ---
    print("\n--- Testing Savitzky-Golay Smoothing ---")
    # Add some NaNs to test handling
    intensity_with_nans = intensity_raw.copy()
    intensity_with_nans[100:110] = np.nan
    smoothed_intensity = smooth_savitzky_golay(
        intensity_with_nans, smoothing_window=15, smoothing_polyorder=3
    )
    print(f"Original mean: {np.nanmean(intensity_with_nans):.2f}, Smoothed mean: {np.nanmean(smoothed_intensity):.2f}")

    # --- Test Noise Analysis ---
    print("\n--- Testing Noise Analysis ---")
    # Use regions away from peaks
    noise_regions = [(200, 250), (400, 450), (750, 800)]
    # Test on baseline-corrected data (SNIP corrected in this case)
    noise_std, regions_used = analyze_noise(wavelengths, intensity_snip_corrected, noise_regions)

    if noise_std is not None:
        print(f"Estimated noise std dev: {noise_std:.4f} (Expected approx 5)")
    else:
        print("Noise analysis failed.")

    # Test with invalid region
    noise_std_invalid, _ = analyze_noise(wavelengths, intensity_snip_corrected, [(900, 950)]) # Region outside data range
    print(f"Noise analysis with out-of-range region returned: {noise_std_invalid}")

    # --- Optional: Plotting results ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))

        plt.subplot(3, 1, 1)
        plt.plot(wavelengths, intensity_raw, label='Raw Intensity', alpha=0.7)
        plt.plot(wavelengths, baseline_true, 'k--', label='True Baseline')
        plt.plot(wavelengths, baseline_poly_calc, 'r-', label='Poly Baseline (O=2, P=5)')
        plt.plot(wavelengths, baseline_snip_calc, 'g-', label='SNIP Baseline (Iter=50)')
        plt.title('Raw Data and Estimated Baselines')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(wavelengths, intensity_poly_corrected, 'r-', label='Poly Corrected')
        plt.plot(wavelengths, intensity_snip_corrected, 'g-', label='SNIP Corrected')
        plt.plot(wavelengths, signal + noise, 'k--', label='True Signal + Noise') # Original signal without baseline
        plt.title('Baseline Corrected Data')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(wavelengths, intensity_with_nans, label='Raw Intensity (with NaNs)', alpha=0.5)
        plt.plot(wavelengths, smoothed_intensity, label=f'Smoothed (SavGol W=15, P=3)', color='purple', linewidth=2)
        plt.title('Smoothing (with NaN handling)')
        plt.legend()
        plt.grid(True)


        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping plots.")
    except Exception as e:
        print(f"\nError during plotting: {e}")