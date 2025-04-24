# -*- coding: utf-8 -*-
"""
Peak detection algorithms for LIBS spectra. Primarily uses SciPy's find_peaks.
Includes options for basic filtering based on height, distance, width, prominence.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# --- SciPy Imports ---
SCIPY_AVAILABLE = False
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    # logging is configured by the main application
    logging.error("SciPy library not found. Peak detection functionality unavailable. Install with 'pip install scipy'.")
    # Dummy function to avoid NameErrors elsewhere, raises error if called
    def find_peaks(*args, **kwargs):
        raise ImportError("SciPy not installed.")

# Import data models and processing helpers
try:
    # Use relative imports assuming standard structure
    from .data_models import Spectrum, Peak
    from .processing import _interpolate_finite # Import the NaN/Inf handler
except ImportError as e_import:
    logging.critical(f"CRITICAL ERROR in peak_detector.py: Cannot import dependencies: {e_import}.")
    raise ImportError(f"Core dependencies failed to import in peak_detector: {e_import}") from e_import


def detect_peaks_scipy(spectrum: Spectrum,
                       rel_height_percent: float = 5.0,
                       min_distance_points: int = 5,
                       min_width_points: Optional[int] = None, # Changed to Optional[int]
                       prominence: Optional[float] = None
                       ) -> List[Peak]:
    """
    Detects peaks in the spectrum's processed intensity using scipy.signal.find_peaks.

    Handles potential NaN/Inf values in the processed data by linear interpolation.

    Args:
        spectrum (Spectrum): The Spectrum object containing processed data.
                             Must have `processed_intensity` attribute set.
        rel_height_percent (float): Minimum peak height relative to the intensity range
                                   (max-min) of the processed data (0-100).
                                   Values outside (0, 100] disable the height filter.
        min_distance_points (int): Minimum required horizontal distance (in data points)
                                  between neighbouring peaks. Must be >= 1.
        min_width_points (Optional[int]): Minimum required width of peaks in data points.
                                          Set to 0 or None to disable. Must be >= 0.
        prominence (Optional[float]): Minimum required vertical distance (prominence) for a peak.
                                      Set to 0 or None to disable. Must be >= 0.

    Returns:
        List[Peak]: A list of detected Peak objects, sorted by wavelength.
                    Returns empty list if no processed data, SciPy unavailable,
                    interpolation fails, or no peaks found matching criteria.
    """
    if not SCIPY_AVAILABLE:
        logging.error("Cannot detect peaks: SciPy library is unavailable.")
        return []

    if spectrum is None or not hasattr(spectrum, 'processed_intensity') or spectrum.processed_intensity is None:
        logging.error("Cannot detect peaks: Processed intensity data is missing or invalid Spectrum object.")
        return []

    if len(spectrum.processed_intensity) < 3: # find_peaks requires at least 3 points implicitly
        logging.warning(f"Cannot detect peaks: Processed spectrum has less than 3 data points ({len(spectrum.processed_intensity)}).")
        return []

    # --- Input Validation and Preparation ---
    x_wavelengths = spectrum.wavelengths # Wavelengths should be clean already
    y_processed = spectrum.processed_intensity

    # Handle potential NaNs/Infs in processed data using interpolation helper
    # This modifies y_search in place potentially
    y_search, finite_mask, interp_ok = _interpolate_finite(y_processed)
    if not interp_ok:
        logging.error("Failed to handle NaNs/Infs in processed data via interpolation. Aborting peak detection.")
        return []
    if not np.all(finite_mask): # Log if interpolation actually happened
        logging.warning("NaNs/Infs were detected and interpolated in processed data before peak detection.")

    # Ensure y_search is 1D (should be guaranteed by Spectrum and _interpolate_finite)
    if y_search.ndim != 1:
         logging.error(f"Peak detection input 'y_search' must be 1D after processing, got shape {y_search.shape}. Aborting.")
         return []

    # Calculate absolute height threshold based on relative percentage of the (potentially interpolated) data
    # Use only originally finite values for range calculation if available, else use interpolated
    y_finite_for_range = y_search[finite_mask] if np.any(finite_mask) else y_search
    if len(y_finite_for_range) == 0:
        logging.error("No finite data points available to calculate peak detection height range. Aborting.")
        return []

    min_intensity = np.min(y_finite_for_range)
    max_intensity = np.max(y_finite_for_range)
    data_range = max_intensity - min_intensity

    height_threshold: Optional[float] = None
    if data_range > 1e-9: # Avoid division by zero or issues with flat lines
        if 0 < rel_height_percent <= 100:
            height_threshold = min_intensity + (rel_height_percent / 100.0) * data_range
        elif rel_height_percent != 0: # Explicitly ignore 0, warn for others outside range
            logging.debug(f"Relative height ({rel_height_percent}%) outside (0, 100]. Disabling height threshold.")
    else:
        logging.warning(f"Data range ({data_range:.2e}) is too small to calculate relative height threshold meaningfully.")

    # Validate distance parameter
    if min_distance_points is None or min_distance_points < 1:
        logging.warning(f"Minimum peak distance ({min_distance_points}) must be >= 1. Using 1.")
        min_distance_points = 1

    # Validate and prepare optional parameters for find_peaks
    find_peaks_kwargs: Dict[str, Any] = {'distance': int(min_distance_points)} # Ensure integer
    if height_threshold is not None:
        find_peaks_kwargs['height'] = height_threshold
    if min_width_points is not None:
        if min_width_points < 0:
             logging.warning(f"Minimum peak width ({min_width_points}) cannot be negative. Disabling width filter.")
        else:
             find_peaks_kwargs['width'] = int(min_width_points) # Already expected int
    if prominence is not None:
         if prominence < 0:
              logging.warning(f"Minimum peak prominence ({prominence}) cannot be negative. Disabling prominence filter.")
         else:
              find_peaks_kwargs['prominence'] = float(prominence) # Ensure float

    logging.info(f"Running scipy.signal.find_peaks with kwargs: {find_peaks_kwargs}")

    # --- Execute find_peaks ---
    try:
        # Use the potentially interpolated y_search data for peak finding
        peak_indices, properties = find_peaks(y_search, **find_peaks_kwargs)
        num_found = len(peak_indices)
        logging.info(f"SciPy find_peaks identified {num_found} raw peak indices.")
        if num_found == 0:
            return []

        # --- Create Peak Objects ---
        detected_peaks: List[Peak] = []
        num_skipped = 0
        for i, idx in enumerate(peak_indices):
            # Ensure index is valid within original spectrum arrays
            if not (0 <= idx < len(x_wavelengths)):
                logging.warning(f"Detected peak index {idx} is out of bounds (spectrum length {len(x_wavelengths)}). Skipping peak {i+1}.")
                num_skipped += 1
                continue

            # Get values at the exact detected index from the *original* processed data
            detected_wavelength = x_wavelengths[idx]
            detected_intensity = y_processed[idx] # Use original processed intensity at peak index

            # Check if detected intensity is valid (could be NaN if input had NaNs before interpolation)
            if not np.isfinite(detected_intensity):
                 logging.warning(f"Peak at index {idx} has non-finite processed intensity ({detected_intensity}) in original data. Skipping peak {i+1}.")
                 num_skipped += 1
                 continue

            # Find corresponding raw intensity (careful about index validity and None)
            raw_intensity_at_peak: Optional[float] = np.nan # Default to NaN
            if spectrum.raw_intensity is not None:
                 if idx < len(spectrum.raw_intensity):
                      raw_val = spectrum.raw_intensity[idx]
                      # Ensure raw value is also finite before storing
                      if np.isfinite(raw_val):
                           raw_intensity_at_peak = float(raw_val)
                      else:
                           logging.debug(f"Raw intensity at peak index {idx} is non-finite ({raw_val}). Storing NaN.")
                 else:
                      # Should not happen if Spectrum class ensures lengths match, but safety check
                      logging.warning(f"Raw intensity array length ({len(spectrum.raw_intensity)}) "
                                      f"mismatch for peak index {idx}. Cannot get raw intensity.")
            else:
                 logging.debug(f"Raw intensity array is None. Cannot get raw intensity for peak index {idx}.")


            # Create Peak object
            try:
                peak_obj = Peak(
                    detected_index=int(idx),
                    detected_wavelength=float(detected_wavelength), # Already checked finite
                    detected_intensity=float(detected_intensity), # Already checked finite
                    raw_intensity_at_peak=raw_intensity_at_peak # Can be NaN
                )
                detected_peaks.append(peak_obj)
            except ValueError as ve: # Catch errors during Peak creation (though unlikely now)
                logging.warning(f"Could not create Peak object for index {idx}: {ve}")
                num_skipped += 1

        # Sort final list of valid peaks by wavelength
        detected_peaks.sort(key=lambda p: p.wavelength_detected)
        log_msg = f"Created {len(detected_peaks)} valid Peak objects."
        if num_skipped > 0:
             log_msg += f" Skipped {num_skipped} invalid peaks."
        logging.info(log_msg)
        return detected_peaks

    except ImportError:
        logging.error("SciPy find_peaks failed: Library not found.")
        return []
    except Exception as e:
        logging.error(f"Error during SciPy peak detection: {e}", exc_info=True)
        return []


# --- Placeholder for NIST-Guided Detection ---
# def detect_peaks_nist_guided(...) -> List[Peak]:
#     logging.warning("NIST-guided peak detection not implemented.")
#     return []