# -*- coding: utf-8 -*-
"""
Core data structures (models) used throughout the application.
These classes help standardize how spectral data, peaks, fits, etc., are represented.
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple, Union, Type # Added Type for from_dict
from dataclasses import dataclass, field, asdict

# --- Constants ---
# Factor to convert sigma (Gaussian std dev) to FWHM
FWHM_GAUSS_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0)) # ~2.35482
# Tolerance for floating point comparisons in __eq__ methods
FLOAT_EQ_TOLERANCE = 1e-7
# Small epsilon for numerical stability checks
EPSILON = 1e-9

# --- Helper Functions ---
def _safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float if possible and finite, else return None."""
    if value is None: return None
    try:
        f_val = float(value)
        return f_val if np.isfinite(f_val) else None
    except (ValueError, TypeError):
        return None

# --- Core Data Models ---

class Spectrum:
    """Represents a single LIBS spectrum."""
    def __init__(self,
                 wavelengths: np.ndarray,
                 raw_intensity: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None,
                 source_filepath: Optional[str] = None):
        """
        Initializes a Spectrum object.

        Args:
            wavelengths (np.ndarray): Array of wavelength values (nm). Should be sorted.
            raw_intensity (np.ndarray): Array of corresponding raw intensity values.
            metadata (Optional[Dict[str, Any]]): Dictionary for metadata.
            source_filepath (Optional[str]): Original file path of the spectrum.

        Raises:
            ValueError: If wavelengths and intensity arrays have mismatched shapes,
                        are not 1D, or are empty.
        """
        if not isinstance(wavelengths, np.ndarray) or not isinstance(raw_intensity, np.ndarray):
            raise TypeError("Wavelengths and Intensity must be NumPy arrays.")
        if wavelengths.shape != raw_intensity.shape:
            raise ValueError(f"Wavelengths ({wavelengths.shape}) and Intensity ({raw_intensity.shape}) arrays must have the same shape.")
        if wavelengths.ndim != 1:
            raise ValueError("Wavelengths and Intensity must be 1-dimensional arrays.")
        if len(wavelengths) == 0:
             raise ValueError("Wavelengths and Intensity arrays cannot be empty.")

        self.wavelengths: np.ndarray = wavelengths
        self.raw_intensity: np.ndarray = raw_intensity
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
        self.source_filepath: Optional[str] = source_filepath

        # --- Processed Data (initialized later by processing steps) ---
        self.processed_intensity: Optional[np.ndarray] = None
        self.baseline: Optional[np.ndarray] = None
        # --- Noise Analysis Data ---
        # These attributes are expected to be populated by external noise analysis routines
        # (e.g., core.processing.analyze_noise called from MainWindow or a dedicated step).
        # Adding a 'calculate_noise' method here would tightly couple data models
        # with specific processing logic and is avoided.
        self.noise_regions: Optional[List[Tuple[float, float]]] = None # List of (start_wl, end_wl) tuples
        self.noise_std_dev: Optional[float] = None # Estimated noise level in signal-free regions

    @property
    def filename(self) -> Optional[str]:
        """Returns the base filename from the source filepath, if available."""
        if self.source_filepath:
            try:
                return os.path.basename(self.source_filepath)
            except Exception:
                return self.source_filepath # Fallback if path manipulation fails
        return None

    def update_processed(self,
                         processed_intensity: np.ndarray,
                         baseline: Optional[np.ndarray] = None):
        """
        Updates the processed intensity and optionally the baseline.

        Args:
            processed_intensity (np.ndarray): The processed intensity array.
            baseline (Optional[np.ndarray]): The calculated baseline array.

        Raises:
            ValueError: If array shapes do not match wavelengths.
        """
        if processed_intensity.shape != self.wavelengths.shape:
            raise ValueError(f"Processed intensity shape {processed_intensity.shape} mismatch with wavelengths {self.wavelengths.shape}.")
        self.processed_intensity = processed_intensity

        if baseline is not None:
            if baseline.shape != self.wavelengths.shape:
                raise ValueError(f"Baseline shape {baseline.shape} mismatch with wavelengths {self.wavelengths.shape}.")
            self.baseline = baseline
        else:
             self.baseline = None # Ensure baseline is cleared if not provided

    def get_data_range(self, wl_min: float, wl_max: float, processed: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns wavelength and intensity data within a specified wavelength range.

        Args:
            wl_min (float): Minimum wavelength.
            wl_max (float): Maximum wavelength.
            processed (bool): If True, returns processed intensity (if available),
                              otherwise returns raw intensity.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Wavelengths and corresponding intensities in the range.
                                          Returns empty arrays if no data in range or source is invalid.
        """
        intensity_source = self.processed_intensity if processed and self.processed_intensity is not None else self.raw_intensity
        if intensity_source is None or self.wavelengths is None:
             logging.warning("Attempted get_data_range with missing intensity or wavelengths.")
             return np.array([]), np.array([])
        try:
             mask = (self.wavelengths >= wl_min) & (self.wavelengths <= wl_max)
             return self.wavelengths[mask], intensity_source[mask]
        except Exception as e:
             logging.error(f"Error getting data range [{wl_min}-{wl_max}]: {e}", exc_info=True)
             return np.array([]), np.array([])

    def __len__(self) -> int:
        """Returns the number of data points in the spectrum."""
        return len(self.wavelengths) if self.wavelengths is not None else 0

    def __repr__(self) -> str:
        try:
            wl_range = f"[{self.wavelengths[0]:.2f}-{self.wavelengths[-1]:.2f}] nm" if len(self) > 0 else "[Empty]"
        except IndexError:
            wl_range = "[Invalid Data]"
        file_info = f", file='{self.filename}'" if self.filename else ""
        proc_info = ", Processed" if self.processed_intensity is not None else ""
        return f"Spectrum(points={len(self)}, range={wl_range}{proc_info}{file_info})"

# --- Consolidated FitResult using @dataclass ---
@dataclass
class FitResult:
    """
    Represents the result of fitting a specific profile to a peak.
    Uses @dataclass for initialization and basic methods.
    """
    # Core fit parameters (no default values - MUST be provided)
    profile_type: str                  # e.g., 'Gaussian', 'Lorentzian', 'PseudoVoigt'
    amplitude: float
    center: float
    width: float                       # Primary width parameter (sigma for Gaussian/PV, gamma HWHM for Lorentzian)

    # Optional parameters / results (with defaults)
    mixing_param_eta: Optional[float] = None # For PseudoVoigt (0=Gauss, 1=Lorentz)
    params_covariance: Optional[np.ndarray] = field(default=None, repr=False) # Covariance matrix from fit
    r_squared: Optional[float] = field(default=None, repr=True)    # Goodness-of-fit metric
    aic: Optional[float] = field(default=None, repr=True)          # Model selection criterion
    bic: Optional[float] = field(default=None, repr=True)          # Model selection criterion
    success: bool = False              # Did the fit converge successfully?
    message: str = ""                  # Optional message from the fitter
    # Data context (optional, useful for plotting/refitting)
    roi_wavelengths: Optional[np.ndarray] = field(default=None, repr=False)
    roi_intensity_corrected: Optional[np.ndarray] = field(default=None, repr=False)
    fitted_curve: Optional[np.ndarray] = field(default=None, repr=False) # Fitted Y values over ROI
    # Peak area (calculated from fit parameters - needs implementation in peak_fitter)
    area: Optional[float] = field(default=None, repr=True) # Integral of the fitted profile
    area_err: Optional[float] = field(default=None, repr=True) # Error estimate for area

    # Post-init calculated fields (calculated after basic initialization)
    fwhm: Optional[float] = field(init=False, default=None, repr=True) # Calculated Full Width at Half Maximum
    param_errors: List[Optional[float]] = field(init=False, default_factory=list, repr=True) # [amp_err, cen_err, wid_err, (eta_err), (area_err?)]

    def __post_init__(self):
        """Calculate FWHM and parameter errors after dataclass initialization."""
        # Ensure calculations happen even if called by from_dict
        if self.fwhm is None:
             self.fwhm = self._calculate_fwhm()
        if not self.param_errors: # Only calculate if list is empty
             self.param_errors = self._calculate_errors()
             # TODO: If area_err calculation is added, append it here?

    def _calculate_fwhm(self) -> Optional[float]:
        """Calculates FWHM based on profile type and width parameter(s) if possible."""
        if not np.isfinite(self.width) or self.width <= EPSILON:
            logging.debug(f"Cannot calculate FWHM: Invalid width ({self.width}) for profile '{self.profile_type}'.")
            return np.nan

        try:
            if self.profile_type == 'Gaussian':
                # width = sigma
                return self.width * FWHM_GAUSS_FACTOR
            elif self.profile_type == 'Lorentzian':
                # width = gamma (HWHM)
                return self.width * 2.0
            elif self.profile_type == 'PseudoVoigt':
                # FWHM for PV is complex. Use approximation (e.g., Thompson, Cox, Hastings 1987)
                # FWHM ≈ (fG^5 + 2.69269*fG^4*fL + 2.42843*fG^3*fL^2 + 4.47163*fG^2*fL^3 + 0.07842*fG*fL^4 + fL^5)^(1/5)
                # where fG = Gaussian FWHM, fL = Lorentzian FWHM
                # Simpler Approximation: Weighted average based on eta (often inaccurate but simple)
                # Approx FWHM ≈ eta * FWHM_Lorentz + (1-eta) * FWHM_Gauss
                # where FWHM_Gauss = sigma * 2.355, FWHM_Lorentz = gamma * 2 = (FWHM_Gauss / 2.355 * 2.355 / 2) * 2 = FWHM_Gauss
                # If 'width' is sigma:
                if self.mixing_param_eta is not None and np.isfinite(self.mixing_param_eta):
                    eta = np.clip(self.mixing_param_eta, 0.0, 1.0)
                    fwhm_g = self.width * FWHM_GAUSS_FACTOR
                    fwhm_l = self.width * 2.0 # Approx Lorentzian FWHM if width were gamma; using sigma here might require adjustment
                    # Let's use a more standard numerical approximation if possible, or document this clearly
                    # For now, report the Gaussian component's FWHM as an approximation
                    logging.warning(f"FWHM calculation for PseudoVoigt ({self.center:.2f}nm) is approximate (based on Gaussian component).")
                    return fwhm_g
                else:
                    logging.warning(f"Cannot calculate PseudoVoigt FWHM: Missing mixing parameter eta.")
                    return np.nan
            # elif self.profile_type == 'Voigt':
                # True Voigt FWHM requires numerical methods or complex approximations based on sigma and gamma.
                # logging.warning("FWHM calculation for true Voigt profile not implemented.")
                # return np.nan
            else:
                logging.warning(f"Cannot calculate FWHM: Unknown profile type '{self.profile_type}'.")
                return np.nan
        except Exception as e:
            logging.error(f"Error calculating FWHM for profile '{self.profile_type}': {e}", exc_info=True)
            return np.nan

    def _calculate_errors(self) -> List[Optional[float]]:
        """Estimates parameter errors (std dev) from the diagonal of the covariance matrix."""
        cov = self.params_covariance
        num_expected_params = 0
        if self.profile_type == 'Gaussian' or self.profile_type == 'Lorentzian':
            num_expected_params = 3 # Amp, Cen, Width
        elif self.profile_type == 'PseudoVoigt':
            num_expected_params = 4 # Amp, Cen, Sigma, Eta
        # Add 'Voigt' if implemented

        default_errors: List[Optional[float]] = [np.nan] * num_expected_params

        if cov is None:
            logging.debug(f"Cannot calculate parameter errors: Covariance matrix is None for profile '{self.profile_type}'.")
            return default_errors
        if not isinstance(cov, np.ndarray):
             logging.warning(f"Cannot calculate parameter errors: Covariance is not a NumPy array (type: {type(cov)}).")
             return default_errors

        # Check shape against expected number of *fitted* parameters
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
             logging.warning(f"Cannot calculate parameter errors: Covariance matrix is not square (shape: {cov.shape}).")
             return default_errors
        # Check if the shape matches the expected number of parameters for the profile type
        # Note: If parameters were fixed during fit, cov matrix might be smaller.
        # The fitter should ideally return errors mapped correctly even if some params were fixed.
        # For now, assume cov matches the expected number of *potentially fitted* parameters.
        if cov.shape[0] != num_expected_params:
             logging.warning(f"Covariance matrix shape ({cov.shape}) does not match expected parameters ({num_expected_params}) "
                             f"for profile '{self.profile_type}'. Parameter errors may be incomplete or incorrect.")
             # Return default errors size if shape mismatch is drastic? Or try to parse?
             # Let's try to parse what's there, padding with NaN.
             default_errors = [np.nan] * cov.shape[0] # Adjust default size


        try:
            diag_variances = np.diag(cov)
            errors = np.full_like(diag_variances, np.nan) # Initialize with NaN

            # Check for invalid variances (negative or infinite)
            valid_mask = (diag_variances >= 0) & np.isfinite(diag_variances)
            invalid_mask = ~valid_mask

            if np.any(invalid_mask):
                 logging.warning(f"Invalid variances found in covariance matrix diagonal for profile '{self.profile_type}'. Errors for these parameters will be NaN.")

            # Calculate sqrt only for valid variances
            errors[valid_mask] = np.sqrt(diag_variances[valid_mask])

            # Pad with NaNs if cov matrix was smaller than expected
            final_errors = errors.tolist()
            if len(final_errors) < num_expected_params:
                 final_errors.extend([np.nan] * (num_expected_params - len(final_errors)))

            return final_errors[:num_expected_params] # Return list of expected size

        except Exception as e:
            logging.error(f"Error calculating parameter errors from covariance for profile '{self.profile_type}': {e}", exc_info=True)
            # Return default errors matching expected length based on profile type
            return [np.nan] * num_expected_params


    def get_param_dict(self, include_errors: bool = False) -> Dict[str, Any]:
         """Returns fit parameters as a dictionary."""
         params = {
             "Fit Profile": self.profile_type,
             "Fitted Amplitude": self.amplitude,
             "Fitted Center (nm)": self.center,
             "Fitted Sigma/Gamma (nm)": self.width, # Clarify this is sigma/gamma
             "Fitted FWHM (nm)": self.fwhm,
             "Fitted Area (a.u.)": self.area, # Added Area
             "Fit R^2": self.r_squared,
             "Fit AIC": self.aic,
             "Fit BIC": self.bic,
         }
         if self.profile_type == 'PseudoVoigt':
              params["Fit Mixing (eta)"] = self.mixing_param_eta
         if include_errors:
              errs = self.param_errors
              params["Fit Amp Error"] = errs[0] if len(errs)>0 else np.nan
              params["Fit Cen Error"] = errs[1] if len(errs)>1 else np.nan
              params["Fit Wid Error"] = errs[2] if len(errs)>2 else np.nan
              # Add eta error if available
              if self.profile_type == 'PseudoVoigt' and len(errs) > 3:
                   params["Fit Eta Error"] = errs[3]
              # Add area error if available (assuming it might be appended later)
              params["Fit Area Error"] = self.area_err # Directly use attribute

         # Clean up NaN/None before returning? Optional.
         # params = {k: v for k, v in params.items() if v is not None and np.isfinite(v)}
         return params

    def __eq__(self, other): # Check for approximate equality
        if not isinstance(other, FitResult): return NotImplemented
        if self.profile_type != other.profile_type: return False

        # Compare core parameters using tolerance
        params_self = [self.amplitude, self.center, self.width, self.mixing_param_eta or 0]
        params_other = [other.amplitude, other.center, other.width, other.mixing_param_eta or 0]
        core_params_equal = np.allclose(params_self, params_other, rtol=FLOAT_EQ_TOLERANCE, atol=FLOAT_EQ_TOLERANCE, equal_nan=True)

        # Optionally compare other fields like success, area etc. if strict equality needed
        # other_fields_equal = (self.success == other.success and
        #                       np.isclose(self.area or np.nan, other.area or np.nan, rtol=FLOAT_EQ_TOLERANCE, atol=FLOAT_EQ_TOLERANCE, equal_nan=True))

        return core_params_equal # & other_fields_equal

    def __hash__(self):
        # Warning: Hash on floats can be problematic. Two results that are
        # 'close enough' via __eq__ might have different hashes if rounding differs.
        # Use with caution in sets/dicts where hash consistency tied to np.allclose is critical.
        # Hashing based on rounded core parameters provides some stability.
        rounded_eta = round(self.mixing_param_eta, 6) if self.mixing_param_eta is not None else 0
        return hash((
            self.profile_type,
            round(self.amplitude, 6),
            round(self.center, 6),
            round(self.width, 6),
            rounded_eta
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes FitResult object to a dictionary suitable for JSON."""
        # Use dataclasses.asdict for basic fields, handle non-serializable manually
        data = asdict(self)
        # Convert NumPy arrays to lists (or None)
        data['params_covariance'] = self.params_covariance.tolist() if isinstance(self.params_covariance, np.ndarray) else None
        data['roi_wavelengths'] = self.roi_wavelengths.tolist() if isinstance(self.roi_wavelengths, np.ndarray) else None
        data['roi_intensity_corrected'] = self.roi_intensity_corrected.tolist() if isinstance(self.roi_intensity_corrected, np.ndarray) else None
        data['fitted_curve'] = self.fitted_curve.tolist() if isinstance(self.fitted_curve, np.ndarray) else None
        # param_errors is already a list of Optional[float] which should be serializable
        # Ensure profile_type is stored as string value if it's an Enum
        if hasattr(self.profile_type, 'value'):
             data['profile_type'] = self.profile_type.value
        # Clean NaNs/Infs for JSON compatibility? Usually handled by custom JSON encoder.
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['FitResult']:
        """Deserializes a FitResult object from a dictionary."""
        if not data or not isinstance(data, dict): return None
        try:
            # Prepare data for dataclass init
            init_data = data.copy()

            # Convert lists back to NumPy arrays if needed
            for key in ['params_covariance', 'roi_wavelengths', 'roi_intensity_corrected', 'fitted_curve']:
                if key in init_data and isinstance(init_data[key], list):
                    init_data[key] = np.array(init_data[key])
                elif key in init_data and init_data[key] is None:
                    init_data[key] = None # Keep as None if saved as null
                else:
                    init_data[key] = None # Default to None if key missing or invalid type

            # Handle potential ProfileType enum (remove if not using Enum)
            # if 'profile_type' in init_data:
            #     try: init_data['profile_type'] = ProfileType(init_data['profile_type'])
            #     except ValueError: logging.warning(f"Invalid profile type '{init_data['profile_type']}' in saved data."); return None

            # Remove fields that are calculated in __post_init__ or not part of __init__
            init_data.pop('fwhm', None)
            init_data.pop('param_errors', None)

            # Create instance using dataclass __init__
            instance = cls(**init_data)
            # __post_init__ will recalculate fwhm and param_errors
            return instance

        except (TypeError, KeyError, ValueError) as e:
            logging.error(f"Failed to deserialize FitResult from dict: {e}. Data: {data}", exc_info=True)
            return None


class Peak:
    """Represents a detected peak in a spectrum, potentially with fit results and identification."""
    def __init__(self,
                 detected_index: int,
                 detected_wavelength: float,
                 detected_intensity: Optional[float], # Make optional for robustness
                 raw_intensity_at_peak: Optional[float] # Make optional for robustness
                 ):
        # Basic validation during init
        if not isinstance(detected_index, int) or detected_index < 0:
             logging.warning(f"Peak created with invalid index: {detected_index}")
             # Assign default or raise error? Assigning a default might hide issues.
             # For now, allow creation but log. Downstream checks needed.
             self.index: int = -1 # Indicate invalid index
        else:
             self.index: int = detected_index

        self.wavelength_detected: float = float(detected_wavelength) if np.isfinite(detected_wavelength) else np.nan
        self.intensity_processed: Optional[float] = float(detected_intensity) if detected_intensity is not None and np.isfinite(detected_intensity) else None
        self.intensity_raw: Optional[float] = float(raw_intensity_at_peak) if raw_intensity_at_peak is not None and np.isfinite(raw_intensity_at_peak) else None

        if self.intensity_processed is None:
             logging.warning(f"Peak created (Idx {self.index}, Wl {self.wavelength_detected:.3f}) with non-finite processed intensity.")
        if self.intensity_raw is None:
             logging.debug(f"Peak created (Idx {self.index}, Wl {self.wavelength_detected:.3f}) with non-finite raw intensity.")

        # --- Analysis Results (populated later) ---
        self.best_fit: Optional[FitResult] = None
        self.alternative_fits: Dict[str, FitResult] = {} # Keyed by profile_type string
        self.potential_matches: List['NISTMatch'] = []

    def add_fit_result(self, fit_result: FitResult, is_best: bool = False):
        """Adds a fitting result, updates best_fit if applicable."""
        if not isinstance(fit_result, FitResult):
            logging.warning("Attempted to add non-FitResult object to peak.")
            return
        # Use profile_type string as key consistently
        profile_key = str(fit_result.profile_type)
        if is_best:
            self.best_fit = fit_result
        self.alternative_fits[profile_key] = fit_result

    def add_nist_match(self, match: 'NISTMatch'):
        """Adds a NIST match if not already present (based on key attributes)."""
        if not isinstance(match, NISTMatch):
            logging.warning("Attempted to add non-NISTMatch object to peak.")
            return
        # Avoid adding duplicate matches (based on element, ion, db_wavelength)
        # Enhance check for robustness
        for existing_match in self.potential_matches:
             match_equal = True
             try:
                 if match.element != existing_match.element: match_equal = False
                 if match.ion_state_str != existing_match.ion_state_str: match_equal = False
                 if not np.isclose(match.wavelength_db, existing_match.wavelength_db, atol=1e-4): match_equal = False
                 # Optionally check Aki or other keys? Keep it simple for now.
             except AttributeError:
                  match_equal = False # Cannot compare if attributes missing
             if match_equal:
                 # logging.debug(f"Duplicate NIST match ignored for peak {self.index}: {match}")
                 return # Already have this match

        self.potential_matches.append(match)
        # Sort matches by proximity to the fitted/detected peak wavelength after adding
        self.potential_matches.sort(key=lambda m: abs(getattr(m, 'wavelength_db', np.inf) - self.wavelength_fitted_or_detected))

    @property
    def wavelength_fitted_or_detected(self) -> float:
        """Returns the fitted center wavelength if available and valid, otherwise the detected."""
        if self.best_fit and self.best_fit.center is not None and np.isfinite(self.best_fit.center):
            return self.best_fit.center
        return self.wavelength_detected

    def to_dataframe_row(self) -> Dict[str, Any]:
        """Converts peak data (including best fit) to a dictionary suitable for DataFrames/Tables."""
        row: Dict[str, Any] = {
            "Peak Index": self.index,
            "Detected Wavelength (nm)": self.wavelength_detected,
            "Raw Intensity": self.intensity_raw,
            "Processed Intensity": self.intensity_processed,
        }
        # Add columns for best fit parameters using FitResult's dict method
        fit_params = self.best_fit.get_param_dict(include_errors=True) if self.best_fit else {}
        row.update(fit_params)

        # Ensure all expected columns exist, even if fit failed (populate with NaN)
        # Make sure this list is comprehensive based on FitResult.get_param_dict keys
        expected_fit_cols = [
            "Fit Profile", "Fitted Center (nm)", "Fitted Amplitude",
            "Fitted Sigma/Gamma (nm)", "Fitted FWHM (nm)", "Fitted Area (a.u.)",
            "Fit Mixing (eta)", "Fit R^2", "Fit AIC", "Fit BIC",
            "Fit Amp Error", "Fit Cen Error", "Fit Wid Error",
            "Fit Eta Error", "Fit Area Error" # Ensure Eta and Area errors are included
        ]
        for col in expected_fit_cols:
            row.setdefault(col, np.nan) # Use NaN for missing fit data

        # Optionally add summary of NIST matches? (e.g., number or best element)
        # row["NIST Matches Count"] = len(self.potential_matches)
        # row["Best NIST Element"] = self.potential_matches[0].element if self.potential_matches else None
        return row

    def __repr__(self) -> str:
        fit_str = f", fit='{self.best_fit.profile_type}'" if self.best_fit else ""
        match_str = f", matches={len(self.potential_matches)}" if self.potential_matches else ""
        return f"Peak(idx={self.index}, wl={self.wavelength_detected:.3f}{fit_str}{match_str})"

    def __eq__(self, other):
        """Equality based on detected index."""
        if not isinstance(other, Peak):
            return NotImplemented
        return self.index == other.index

    def __hash__(self):
        """Hash based on detected index."""
        return hash(self.index)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes Peak object to a dictionary suitable for JSON."""
        return {
            'detected_index': self.index,
            'detected_wavelength': self.wavelength_detected,
            'detected_intensity': self.intensity_processed,
            'raw_intensity_at_peak': self.intensity_raw,
            'best_fit': self.best_fit.to_dict() if self.best_fit else None,
            'alternative_fits': {prof: fit.to_dict() for prof, fit in self.alternative_fits.items()},
            'potential_matches': [match.to_dict() for match in self.potential_matches]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['Peak']:
        """Deserializes a Peak object from a dictionary."""
        if not data or not isinstance(data, dict): return None
        try:
            peak = cls(
                detected_index=data['detected_index'],
                detected_wavelength=data['detected_wavelength'],
                detected_intensity=data.get('detected_intensity'), # Use .get() for optional fields
                raw_intensity_at_peak=data.get('raw_intensity_at_peak')
            )

            if 'best_fit' in data and data['best_fit'] is not None:
                peak.best_fit = FitResult.from_dict(data['best_fit'])

            if 'alternative_fits' in data and isinstance(data['alternative_fits'], dict):
                for prof, fit_data in data['alternative_fits'].items():
                    fit_obj = FitResult.from_dict(fit_data)
                    if fit_obj:
                        peak.alternative_fits[prof] = fit_obj

            if 'potential_matches' in data and isinstance(data['potential_matches'], list):
                for match_data in data['potential_matches']:
                    match_obj = NISTMatch.from_dict(match_data)
                    if match_obj:
                        peak.potential_matches.append(match_obj)
                # Re-sort matches after loading (optional, but good practice)
                peak.potential_matches.sort(key=lambda m: abs(getattr(m, 'wavelength_db', np.inf) - peak.wavelength_fitted_or_detected))

            return peak
        except (KeyError, TypeError, ValueError) as e:
            logging.error(f"Failed to deserialize Peak from dict: {e}. Data: {data}", exc_info=True)
            return None


class NISTMatch:
    """
    Represents a potential match from the NIST database for a spectral line.

    Note: For convenience in Boltzmann plots, this class stores the NIST upper level
          energy (Ek) and stat. weight (gk) as `self.ei` and `self.gi` respectively.
          The lower level E/g are stored as `self.ek` and `self.gk`.
    """
    def __init__(self,
                 element: str,
                 ion_state_str: str,
                 wavelength_db: float,
                 aki: Optional[float] = None, # Transition Probability (A_ki) s^-1
                 ei: Optional[float] = None,  # Upper energy level (E_k from NIST) in eV
                 gi: Optional[float] = None,  # Upper statistical weight (g_k from NIST)
                 ek: Optional[float] = None,  # Lower energy level (E_i from NIST) in eV
                 gk: Optional[float] = None,  # Lower statistical weight (g_i from NIST)
                 line_label: Optional[str] = None, # Optional descriptive label
                 source: str = 'Unknown'      # Source of the match (e.g., 'NIST Online')
                 ):
        self.element: str = str(element) if element else "Unknown"
        self.ion_state_str: str = str(ion_state_str) if ion_state_str else "?"
        self.wavelength_db: float = float(wavelength_db) if np.isfinite(wavelength_db) else np.nan

        # Use helper for safe float conversion
        self.aki: Optional[float] = _safe_float(aki)
        self.ei: Optional[float] = _safe_float(ei)  # Upper E
        self.gi: Optional[float] = _safe_float(gi)  # Upper g
        self.ek: Optional[float] = _safe_float(ek)  # Lower E
        self.gk: Optional[float] = _safe_float(gk)  # Lower g

        # Generate a safer default label if none provided
        if line_label:
             self.line_label: str = str(line_label)
        else:
            wl_str = f"{self.wavelength_db:.3f}" if np.isfinite(self.wavelength_db) else "N/A"
            self.line_label: str = f"{self.element} {self.ion_state_str} @ {wl_str}"

        self.source: str = str(source)

        # Attributes to link back to the peak that generated this match query
        self.query_peak_index: Optional[int] = None
        self.query_peak_wavelength: Optional[float] = None

    @property
    def ion_state_int(self) -> Optional[int]:
        """Converts Roman numeral ion state string to integer (I=1, II=2, etc.)."""
        # Extend map if higher states needed
        roman_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8}
        return roman_map.get(self.ion_state_str.upper())

    def to_dataframe_row(self, peak_wavelength: float, peak_intensity: float) -> Dict[str, Any]:
        """
        Converts match data to a dictionary suitable for the NIST results table display.
        Uses clearer column names for upper/lower states.
        """
        delta_lambda = peak_wavelength - self.wavelength_db if np.isfinite(peak_wavelength) and np.isfinite(self.wavelength_db) else np.nan
        # Match columns expected by NistSearchView table
        return {
            "Peak λ (nm)": peak_wavelength,     # Wavelength of the peak this matches
            "Intensity": peak_intensity,        # Intensity of the peak this matches
            "Source": self.source,              # Where the match came from
            "Elem": self.element,               # Element symbol
            "Ion": self.ion_state_str,          # Ionization state string (e.g., "I", "II")
            "DB λ (nm)": self.wavelength_db,    # Database wavelength
            "Δλ (nm)": delta_lambda,            # Difference: Peak Wl - DB Wl
            "Aki (s⁻¹)": self.aki,              # Transition Probability A_ki
            "E_upper (eV)": self.ei,            # **Upper** Energy Level (NIST E_k)
            "g_upper": self.gi,                 # **Upper** Stat. Weight (NIST g_k)
            "E_lower (eV)": self.ek,            # **Lower** Energy Level (NIST E_i)
            "g_lower": self.gk,                 # **Lower** Stat. Weight (NIST g_i)
            "Line Label": self.line_label,      # Descriptive label
        }

    def __repr__(self) -> str:
        return f"NISTMatch({self.element} {self.ion_state_str} @ {self.wavelength_db:.4f} nm, Src={self.source})"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes NISTMatch object to a dictionary suitable for JSON."""
        # Use vars() or manual creation for simplicity, no complex types here
        return vars(self) # Returns dictionary of instance attributes

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['NISTMatch']:
        """Deserializes a NISTMatch object from a dictionary."""
        if not data or not isinstance(data, dict): return None
        try:
            # Pass dictionary directly to __init__
            instance = cls(**data)
            return instance
        except (TypeError, KeyError, ValueError) as e:
            logging.error(f"Failed to deserialize NISTMatch from dict: {e}. Data: {data}", exc_info=True)
            return None