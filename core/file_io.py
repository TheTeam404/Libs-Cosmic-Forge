"""
Functions for loading and saving spectral data and analysis results.
Handles various text formats and saving to CSV.
"""

import os
import logging
import traceback
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

# --- Placeholder Data Models ---
# !! IMPORTANT: Replace these with imports from your actual data_models module !!
# Example: from .data_models import Spectrum, Peak
class Peak:
    """ Placeholder Peak class. Replace with your actual implementation. """
    def __init__(self, index, detected_wave, raw_int, proc_int=None, fit_results=None, fit_errors=None):
        self.index = index
        self.wavelength_detected = detected_wave # Using detected for simplicity here
        self.raw_intensity_at_detected = raw_int
        self.processed_intensity_at_detected = proc_int

        # --- Fit Results ---
        # Safely get attributes from fit_results dict if it exists
        self.best_fit = None # Placeholder for the FitResult object if you use one
        self.fit_profile = fit_results.get('profile', None) if fit_results else None
        self.fit_center = fit_results.get('center', None) if fit_results else None
        self.fit_amplitude = fit_results.get('amplitude', None) if fit_results else None
        self.fit_width = fit_results.get('sigma', None) if fit_results else None # Assuming sigma represents width
        self.fit_fwhm = fit_results.get('fwhm', None) if fit_results else None
        self.fit_eta = fit_results.get('eta', None) if fit_results else None # Mixing
        self.fit_r_squared = fit_results.get('r_squared', None) if fit_results else None
        self.fit_aic = fit_results.get('aic', None) if fit_results else None
        self.fit_bic = fit_results.get('bic', None) if fit_results else None

        # --- Fit Errors ---
        self.fit_amplitude_error = fit_errors.get('amplitude', None) if fit_errors else None
        self.fit_center_error = fit_errors.get('center', None) if fit_errors else None
        self.fit_width_error = fit_errors.get('sigma', None) if fit_errors else None

        # --- Added for compatibility ---
        self.potential_matches = [] # List to hold potential NISTMatch objects

    @property
    def wavelength_fitted_or_detected(self) -> Optional[float]:
        """Returns fitted wavelength if available and valid, otherwise detected."""
        if self.fit_center is not None and np.isfinite(self.fit_center):
            return self.fit_center
        # Ensure detected wavelength is valid before returning
        if self.wavelength_detected is not None and np.isfinite(self.wavelength_detected):
             return self.wavelength_detected
        return None # Return None if neither is valid

    @property
    def intensity_fitted(self) -> Optional[float]:
         """Returns fitted amplitude if available."""
         # Ensure amplitude is valid before returning
         if self.fit_amplitude is not None and np.isfinite(self.fit_amplitude):
              return self.fit_amplitude
         return None

    def to_dataframe_row(self) -> Dict[str, Any]:
        """ Returns a dictionary representation suitable for a DataFrame row. """
        # This method MUST exist in your actual Peak class for save_peak_list to work
        return {
            "Peak Index": self.index,
            "Detected Wavelength (nm)": self.wavelength_detected,
            "Raw Intensity": self.raw_intensity_at_detected,
            "Processed Intensity": self.processed_intensity_at_detected,
            "Fit Profile": self.fit_profile,
            "Fitted Center (nm)": self.fit_center,
            "Fitted Amplitude": self.fit_amplitude,
            "Fitted Width (nm)": self.fit_width, # Often Sigma for Gaussian/Voigt
            "Fitted FWHM (nm)": self.fit_fwhm, # Full Width at Half Maximum
            "Fit Mixing (eta)": self.fit_eta, # For Pseudo-Voigt
            "Fit R^2": self.fit_r_squared,
            "Fit AIC": self.fit_aic,
            "Fit BIC": self.fit_bic,
            "Fit Amp Error": self.fit_amplitude_error,
            "Fit Cen Error": self.fit_center_error,
            "Fit Wid Error": self.fit_width_error,
            # Add other relevant fields like correlation info if needed
             "Correlated Matches": len(self.potential_matches) if hasattr(self, 'potential_matches') else 0
        }

class Spectrum:
    """ Placeholder Spectrum class. Replace with your actual implementation. """
    def __init__(self, wavelengths: np.ndarray, raw_intensity: np.ndarray, metadata: Dict = None, source_filepath: str = None, processed_intensity: np.ndarray = None, baseline: np.ndarray = None):
        # Basic validation on initialization
        if wavelengths is None or raw_intensity is None:
            raise ValueError("Wavelengths and raw_intensity cannot be None.")
        if not isinstance(wavelengths, np.ndarray) or not isinstance(raw_intensity, np.ndarray):
            raise TypeError("Wavelengths and raw_intensity must be NumPy arrays.")
        if wavelengths.ndim != 1 or raw_intensity.ndim != 1:
             raise ValueError("Wavelengths and raw_intensity must be 1-dimensional arrays.")
        if wavelengths.shape != raw_intensity.shape:
             raise ValueError(f"Wavelengths shape {wavelengths.shape} must match Raw Intensity shape {raw_intensity.shape}")

        self.wavelengths = wavelengths
        self.raw_intensity = raw_intensity
        self.metadata = metadata if metadata is not None else {}
        self.filename = source_filepath # Use 'filename' consistent with MainWindow usage
        self.processed_intensity = processed_intensity # Can be None initially
        self.baseline = baseline # Can be None initially
        self.peaks: List[Peak] = [] # Initialize empty list of peaks

    def __len__(self):
        """Return number of data points."""
        return len(self.wavelengths) # Assumes wavelengths is always a valid array after init

    def update_processed(self, processed_intensity: np.ndarray, baseline: np.ndarray):
         """Updates processed data and baseline, ensuring dimensions match."""
         if not isinstance(processed_intensity, np.ndarray) or not isinstance(baseline, np.ndarray):
              raise TypeError("Processed intensity and baseline must be NumPy arrays.")
         if processed_intensity.shape != self.wavelengths.shape or baseline.shape != self.wavelengths.shape:
             logging.error(f"Length mismatch during update_processed: wl={self.wavelengths.shape}, proc={processed_intensity.shape}, base={baseline.shape}")
             raise ValueError("Processed/baseline data shape must match wavelength shape.")
         self.processed_intensity = processed_intensity
         self.baseline = baseline
         logging.debug(f"Spectrum '{os.path.basename(self.filename or 'Unknown')}' processed data updated.")

    def __str__(self):
        proc_info = f", Processed" if self.processed_intensity is not None else ""
        base_info = f", Baseline" if self.baseline is not None else ""
        peak_info = f", {len(self.peaks)} Peaks" if self.peaks else ""
        return f"Spectrum({os.path.basename(self.filename or 'Unknown')}, Pts: {len(self)}{proc_info}{base_info}{peak_info})"
# --- End Placeholder Data Models ---


# --- Constants ---
DEFAULT_SAVE_DELIMITER = ',' # Use comma for CSV saving by default
DEFAULT_FLOAT_FORMAT = '%.6g' # Use general format with up to 6 significant digits for saving
DEFAULT_COMMENT_CHAR = '#' # Default character for comments if none specified

# --- Loading Function ---
def load_spectrum_from_file(filepath: str,
                            delimiter: Optional[str] = None,
                            comment_char: Optional[str] = DEFAULT_COMMENT_CHAR, # Changed name, provide default
                            encoding: str = 'utf-8') -> Spectrum:
    """
    Loads spectral data from a text-based file into a Spectrum object.

    Handles common delimiters (tab, comma, space, semicolon) and comment lines.
    Attempts to coerce data to numeric types and handles basic errors.
    Sorts data by wavelength and removes duplicate wavelengths.

    Args:
        filepath (str): The full path to the data file.
        delimiter (Optional[str]): The column delimiter. If None, attempts common ones.
        comment_char (Optional[str]): Character indicating comment lines to ignore.
                                      Defaults to '#'. Use None or '' for no comment character.
        encoding (str): File encoding (default: 'utf-8').

    Returns:
        Spectrum: An initialized Spectrum object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid, data cannot be parsed, or < 2 valid points remain.
        Exception: For other unexpected errors during loading.
    """
    logging.info(f"Attempting load: {filepath} (Delimiter: {delimiter or 'Auto'}, Comment: '{comment_char}')")
    if not os.path.exists(filepath): raise FileNotFoundError(f"File not found: {filepath}")

    # Use provided comment char, or None if it's an empty string (Pandas treats None/empty differently)
    used_comment = comment_char if comment_char else None
    guessed_delimiter = delimiter # Store original or None

    try:
        # --- Delimiter Guessing (if not provided) ---
        if delimiter is None:
            logging.debug(f"No delimiter provided, attempting to guess for {os.path.basename(filepath)}.")
            try: # Nested try for guessing logic
                with open(filepath, 'r', encoding=encoding) as f:
                    # Read a few lines, skipping potential comment lines
                    lines_to_check = []
                    for _ in range(10): # Read up to 10 lines
                        line = f.readline()
                        if not line: break # End of file
                        line_strip = line.strip()
                        # Check if line is not empty AND (no comment char defined OR line doesn't start with it)
                        if line_strip and (not used_comment or not line_strip.startswith(used_comment)):
                             lines_to_check.append(line_strip)
                        if len(lines_to_check) >= 5: break # Got enough non-comment lines

                if not lines_to_check:
                    raise ValueError("File appears empty or contains only comment lines. Cannot guess delimiter.")

                first_data_line = lines_to_check[0]
                logging.debug(f"First data line for guessing: '{first_data_line}'")

                # Simple checks for common delimiters, prioritizing scientific ones
                if '\t' in first_data_line: guessed_delimiter = '\t'
                elif ',' in first_data_line: guessed_delimiter = ','
                elif ';' in first_data_line: guessed_delimiter = ';'
                elif ' ' in first_data_line:
                     # Check if space likely separates only two numeric fields
                     parts = first_data_line.split()
                     try:
                         # Allow for more than 2 parts, but check if first 2 look numeric
                         if len(parts) >= 2 and all('.' in p or p.isdigit() or ('e' in p.lower()) or ('-' in p) for p in parts[:2]): # Basic numeric check
                             guessed_delimiter = r'\s+' # Use regex for one or more spaces
                         else: raise ValueError("Space separated but not clearly numeric pairs")
                     except ValueError:
                          logging.debug("Space found, but parts don't look like simple numeric pairs. Falling back.")
                          guessed_delimiter = '\t' # Fallback if space doesn't seem right
                else:
                    guessed_delimiter = '\t' # Fallback if no common delimiter detected
                    logging.warning(f"Could not reliably guess delimiter for {os.path.basename(filepath)}, defaulting to Tab ('\\t').")

                logging.info(f"Guessed delimiter: {repr(guessed_delimiter)} for {os.path.basename(filepath)}")

            except Exception as guess_e:
                 logging.warning(f"Delimiter guessing failed ({guess_e}), defaulting to Tab ('\\t').")
                 guessed_delimiter = '\t' # Default fallback

        # --- Read using Pandas ---
        logging.debug(f"Reading with Pandas: sep={repr(guessed_delimiter)}, comment={repr(used_comment)}")
        try:
            # Use engine='python' for flexibility with regex delimiters and comments
            df = pd.read_csv(
                filepath,
                sep=guessed_delimiter, # Use guessed or provided delimiter
                header=None, # Assume no header row
                names=["Wavelength", "Intensity"], # Assign column names
                comment=used_comment, # Pass the comment character
                encoding=encoding,
                skipinitialspace=True, # Handle potential spaces after delimiter
                skip_blank_lines=True, # Ignore empty lines
                on_bad_lines='warn', # Warn about lines with wrong number of fields, don't error immediately
                #low_memory=False, # Recommended for potentially mixed types
                dtype=str, # Read columns as strings initially
                engine='python' # Python engine needed for regex sep and robust comment handling
            )
        except pd.errors.EmptyDataError:
             raise ValueError("File is empty or contains only comments/blank lines.") from None
        except pd.errors.ParserError as pe:
             logging.error(f"Pandas parsing error: {pe}. Check delimiter ('{repr(guessed_delimiter)}'), comments ('{used_comment}'), and file structure.")
             raise ValueError(f"Failed to parse file structure. Check delimiter/comments. Error: {pe}") from pe
        except Exception as read_e: # Catch other potential read errors
             logging.error(f"Error reading file with Pandas: {read_e}", exc_info=True)
             raise IOError(f"Could not read file {os.path.basename(filepath)}.") from read_e


        if df.empty: raise ValueError("No data could be parsed from the file (potentially only comments/blank/bad lines).")
        original_rows = len(df)
        logging.debug(f"Read {original_rows} lines initially (including potentially bad lines if warned).")

        # --- Data Cleaning and Conversion ---
        # Attempt conversion to numeric, coercing errors to NaN
        # Replace comma decimal separator BEFORE converting to numeric
        df['Wavelength'] = pd.to_numeric(df['Wavelength'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df['Intensity'] = pd.to_numeric(df['Intensity'].astype(str).str.replace(',', '.', regex=False), errors='coerce')

        # Drop rows where *either* Wavelength or Intensity failed conversion (became NaN)
        initial_nan_count = df.isnull().any(axis=1).sum()
        if initial_nan_count > 0:
             logging.warning(f"Found {initial_nan_count} rows with non-numeric values. Dropping them.")
             df.dropna(subset=['Wavelength', 'Intensity'], inplace=True)
        rows_after_dropna = len(df)

        if df.empty: raise ValueError(f"No valid numeric Wavelength/Intensity pairs found after parsing and cleaning {original_rows} initial lines.")

        # --- Extract NumPy Arrays and Check Finite ---
        wavelengths = df['Wavelength'].to_numpy(dtype=float)
        raw_intensity = df['Intensity'].to_numpy(dtype=float)

        finite_mask = np.isfinite(wavelengths) & np.isfinite(raw_intensity)
        if not np.all(finite_mask): # Check if any non-finite values (Inf/NaN) remain
            num_nonfinite = (~finite_mask).sum()
            logging.warning(f"Found and removing {num_nonfinite} non-finite values (Inf/NaN) post-conversion.");
            wavelengths = wavelengths[finite_mask]
            raw_intensity = raw_intensity[finite_mask]
        final_finite_rows = len(wavelengths)

        if final_finite_rows < 2:
             raise ValueError(f"Spectrum must contain at least two valid finite data points. Found {final_finite_rows} after cleaning.")

        # --- Sorting and Deduplication ---
        logging.debug("Sorting data by wavelength...")
        sort_indices = np.argsort(wavelengths)
        # Check if already sorted before actually sorting to avoid unnecessary work/logging
        if not np.array_equal(sort_indices, np.arange(len(wavelengths))):
            logging.warning(f"Wavelength data in {os.path.basename(filepath)} was not sorted. Sorting now.")
            wavelengths = wavelengths[sort_indices]
            raw_intensity = raw_intensity[sort_indices]

        logging.debug("Checking for duplicate wavelengths...")
        unique_wavelengths, unique_indices = np.unique(wavelengths, return_index=True)
        final_unique_rows = len(unique_wavelengths)
        if final_unique_rows < len(wavelengths):
            num_dupes = len(wavelengths) - final_unique_rows
            logging.warning(f"Found and removing {num_dupes} duplicate wavelength values (keeping first occurrence).")
            wavelengths = wavelengths[unique_indices]
            raw_intensity = raw_intensity[unique_indices]

        if final_unique_rows < 2:
             raise ValueError(f"Spectrum has less than 2 unique valid points after cleaning and deduplication.")

        # --- Create Spectrum Object ---
        metadata = {
            "original_filename": os.path.basename(filepath),
            "load_timestamp": pd.Timestamp.now().isoformat(),
            "initial_rows_read": original_rows,
            "rows_after_conversion_drop": rows_after_dropna,
            "rows_after_finite_filter": final_finite_rows,
            "final_points": final_unique_rows,
            "guessed_delimiter": repr(guessed_delimiter), # Use repr for clarity (e.g., shows '\t')
            "used_comment_char": repr(used_comment)
        }
        # Use the actual Spectrum class (imported or placeholder)
        spectrum = Spectrum(
            wavelengths=wavelengths,
            raw_intensity=raw_intensity,
            metadata=metadata,
            source_filepath=filepath # Store full path in 'filename' attribute
        )
        logging.info(f"Successfully loaded: {spectrum}")
        return spectrum

    except FileNotFoundError: # Re-raise specific error for calling code
        logging.error(f"File not found during load: {filepath}")
        raise
    except ValueError as ve: # Catch data format/parsing errors
        logging.error(f"Value error loading {filepath}: {ve}")
        # Provide a more user-friendly message potentially
        raise ValueError(f"Invalid format or data in {os.path.basename(filepath)}: {ve}") from ve
    except Exception as e: # Catch other unexpected errors
        logging.error(f"Unexpected error loading {filepath}: {e}", exc_info=True)
        raise Exception(f"Failed to load {os.path.basename(filepath)} due to unexpected error: {e}") from e


# --- Saving Functions ---

def save_spectrum_data(spectrum: Spectrum, filepath: str, include_processed: bool = True, delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """
    Saves spectrum data (wavelength, raw intensity, optionally processed and baseline) to a file.

    Args:
        spectrum (Spectrum): The Spectrum object to save.
        filepath (str): The full path for the output file (e.g., 'output/processed_spectrum.csv').
        include_processed (bool): If True and processed data exists, include it as a column.
        delimiter (str): The delimiter to use for saving (e.g., ',', '\t').
        float_format (str): The format string for saving floating-point numbers.
    """
    if not spectrum or len(spectrum) == 0:
         logging.warning(f"Attempted to save empty or invalid spectrum. Skipping save to {filepath}.")
         return

    logging.info(f"Saving spectrum data ({len(spectrum)} points) to: {filepath}")
    try:
        # Ensure directory exists
        if filepath:
             os.makedirs(os.path.dirname(filepath), exist_ok=True)
        else: raise ValueError("Filepath cannot be empty for saving.")

        data_dict: Dict[str, np.ndarray] = {
            "Wavelength_nm": spectrum.wavelengths,
            "Raw_Intensity": spectrum.raw_intensity
        }
        columns = ["Wavelength_nm", "Raw_Intensity"]

        # Include processed data if requested and valid
        if include_processed and spectrum.processed_intensity is not None:
            if spectrum.processed_intensity.shape == spectrum.wavelengths.shape:
                data_dict["Processed_Intensity"] = spectrum.processed_intensity
                columns.append("Processed_Intensity")
                # Also include baseline if processed is included and baseline exists and matches shape
                if spectrum.baseline is not None and spectrum.baseline.shape == spectrum.wavelengths.shape:
                     data_dict["Baseline"] = spectrum.baseline # Optionally save baseline too
                     columns.append("Baseline")
                     logging.debug("Including baseline in saved spectrum data.")
                else:
                     if spectrum.baseline is not None: # Log only if baseline exists but mismatched
                          logging.warning(f"Baseline length ({spectrum.baseline.shape}) mismatches wavelength length ({spectrum.wavelengths.shape}), baseline NOT saved.")
            else:
                logging.warning(f"Processed intensity shape ({spectrum.processed_intensity.shape}) "
                                f"mismatches wavelength shape ({spectrum.wavelengths.shape}). Processed data NOT saved.")

        df_to_save = pd.DataFrame(data_dict)

        # Use specific columns order
        df_to_save = df_to_save[columns]

        df_to_save.to_csv(
            filepath,
            sep=delimiter,
            header=True, # Include header row
            index=False, # Do not write row indices
            float_format=float_format, # Control float precision
            encoding='utf-8',
            na_rep='NaN', # How to represent Not a Number
            lineterminator='\n' # Explicit line terminator for consistency
        )
        logging.info(f"Spectrum data saved successfully to {filepath}")
    except (ValueError, TypeError, AttributeError, IOError, PermissionError, Exception) as e:
        logging.error(f"Failed to save spectrum data to {filepath}: {e}", exc_info=True)
        raise # Re-raise the exception after logging


def save_peak_list(peaks: List[Peak], filepath: str, delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """
    Saves the detected/fitted peak list to a CSV file.

    Relies on the `Peak.to_dataframe_row()` method existing and returning a dict.

    Args:
        peaks (List[Peak]): The list of Peak objects to save.
        filepath (str): The full path for the output CSV file.
        delimiter (str): The delimiter for the CSV file.
        float_format (str): Format string for floating-point numbers.
    """
    logging.info(f"Attempting to save peak list ({len(peaks)} peaks) to: {filepath}")
    # Define desired output columns explicitly - adjust if your Peak model differs
    # Make sure these keys match the dictionary returned by Peak.to_dataframe_row()
    output_columns = [
        "Peak Index", "Detected Wavelength (nm)", "Raw Intensity", "Processed Intensity",
        "Fit Profile", "Fitted Center (nm)", "Fitted Amplitude", "Fitted Width (nm)",
        "Fitted FWHM (nm)", "Fit Mixing (eta)", "Fit R^2", "Fit AIC", "Fit BIC",
        "Fit Amp Error", "Fit Cen Error", "Fit Wid Error", "Correlated Matches"
    ]

    # Handle empty peak list case
    if not peaks:
        logging.warning("Peak list is empty. Saving file with header only.")
        try:
            # Ensure directory exists
            if filepath:
                 os.makedirs(os.path.dirname(filepath), exist_ok=True)
            else: raise ValueError("Filepath cannot be empty for saving empty peak list.")
            # Create file with header
            pd.DataFrame(columns=output_columns).to_csv(
                 filepath,
                 sep=delimiter,
                 index=False,
                 encoding='utf-8',
                 lineterminator='\n'
            )
            logging.info(f"Saved empty peak list file with header to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save empty peak list header to {filepath}: {e}", exc_info=True)
            raise # Re-raise after logging
        return # Exit function after handling empty list

    # Handle non-empty peak list
    try:
        # Use the Peak object's method to get data for each peak
        peak_data_list = []
        successful_conversions = 0
        for i, peak in enumerate(peaks):
             # Check if the peak object itself is valid and has the method
             if peak and hasattr(peak, 'to_dataframe_row') and callable(peak.to_dataframe_row):
                 try:
                     row_data = peak.to_dataframe_row()
                     if isinstance(row_data, dict):
                         peak_data_list.append(row_data)
                         successful_conversions += 1
                     else:
                          logging.warning(f"Peak {i} 'to_dataframe_row' did not return a dictionary. Skipping.")
                 except Exception as row_e:
                      logging.error(f"Error calling 'to_dataframe_row' for peak {i}: {row_e}. Skipping.", exc_info=False) # Keep log cleaner
             else:
                 logging.error(f"Peak object at index {i} is invalid or missing the 'to_dataframe_row' method. Skipping.")

        if not peak_data_list: # Check if all peaks were skipped
             logging.error("No valid peak data could be extracted (missing/failing 'to_dataframe_row'?). Cannot save peak list.")
             # Optionally save empty file header here too?
             return

        peak_df = pd.DataFrame(peak_data_list)
        logging.debug(f"Created DataFrame with {len(peak_df)} rows from peaks.")

        # Ensure all desired output columns exist in the DataFrame, add if missing
        missing_cols_added = []
        for col in output_columns:
             if col not in peak_df.columns:
                 missing_cols_added.append(col)
                 peak_df[col] = np.nan # Add missing column filled with NaN
        if missing_cols_added:
            logging.warning(f"Added missing columns to peak save data: {', '.join(missing_cols_added)}")

        # Select and order columns for saving - use only columns that ACTUALLY exist in DF to prevent KeyError
        final_columns = [col for col in output_columns if col in peak_df.columns]
        if len(final_columns) != len(output_columns):
            logging.warning(f"Some expected columns were not generated by 'to_dataframe_row' and won't be saved: {set(output_columns) - set(final_columns)}")

        peak_df_to_save = peak_df[final_columns]

        # Ensure directory exists (redundant check, but safe)
        if filepath:
             os.makedirs(os.path.dirname(filepath), exist_ok=True)
        else: raise ValueError("Filepath cannot be empty for saving peak list.")

        # Save the DataFrame
        peak_df_to_save.to_csv(
            filepath,
            sep=delimiter,
            header=True,
            index=False,
            float_format=float_format,
            encoding='utf-8',
            na_rep='NaN', # Representation for NaN values
            lineterminator='\n' # Explicit line terminator
        )
        logging.info(f"Peak list ({len(peak_df_to_save)} peaks) saved successfully to {filepath}")

    except (AttributeError, TypeError, ValueError, IOError, PermissionError, Exception) as e:
        logging.error(f"Failed to save peak list to {filepath}: {e}", exc_info=True)
        raise # Re-raise other exceptions after logging


def save_nist_matches(matches_df: Optional[pd.DataFrame], filepath: str, delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """ Saves the NIST match results DataFrame (if valid) to a CSV file. """
    # Check if matches_df is a DataFrame before passing to generic saver
    if not isinstance(matches_df, pd.DataFrame) and matches_df is not None:
        logging.error("Invalid data type passed to save_nist_matches. Expected pandas DataFrame or None.")
        # Decide if you want to try converting or just fail
        raise TypeError("Data passed to save_nist_matches must be a pandas DataFrame or None.")

    save_dataframe(matches_df, filepath, delimiter, float_format, "NIST matches")


def save_dataframe(df: Optional[pd.DataFrame], filepath: str, delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT, data_description: str = "data"):
    """Generic function to save any pandas DataFrame to CSV with error handling."""
    desc_cap = data_description.capitalize()
    is_valid_df = isinstance(df, pd.DataFrame)
    shape_info = f"({df.shape[0]}x{df.shape[1]})" if is_valid_df else "(None or Empty)"
    logging.info(f"Saving {data_description} {shape_info} to: {filepath}")

    # Ensure directory exists before attempting to save anything
    try:
        if filepath: # Ensure filepath is not empty or None
             # Use os.path.abspath to handle relative paths robustly before getting dirname
             abs_filepath = os.path.abspath(filepath)
             dir_name = os.path.dirname(abs_filepath)
             if dir_name: # Ensure dirname is not empty (e.g., if filepath is just 'file.csv')
                 os.makedirs(dir_name, exist_ok=True)
        else:
             logging.error(f"Cannot save {data_description}: filepath is empty or None.")
             raise ValueError(f"Invalid filepath provided for saving {data_description}.")

        # Handle None or empty DataFrame
        if not is_valid_df or df.empty:
            log_msg = f"{desc_cap} DataFrame is None." if df is None else f"{desc_cap} DataFrame is empty."
            logging.warning(f"{log_msg} Saving file with header only (if columns available).")
            # Define columns: use df columns if it's an empty df, otherwise empty list
            columns_to_write = df.columns if is_valid_df and not df.columns.empty else []
            # Create empty DF with columns and save
            pd.DataFrame(columns=columns_to_write).to_csv(
                filepath,
                sep=delimiter,
                index=False,
                encoding='utf-8',
                lineterminator='\n'
            )
            logging.info(f"Saved empty {data_description} file (with header if available) to {filepath}.")
            return # Exit function

        # Save non-empty DataFrame
        df.to_csv(
            filepath,
            sep=delimiter,
            header=True,
            index=False,
            float_format=float_format,
            encoding='utf-8',
            na_rep='NaN', # Representation for NaN values
            lineterminator='\n' # Explicit line terminator
        )
        logging.info(f"{desc_cap} saved successfully ({df.shape[0]} rows) to {filepath}")

    except (ValueError, TypeError, IOError, PermissionError, Exception) as e:
        logging.error(f"Failed to save {data_description} to {filepath}: {e}", exc_info=True)
        raise # Re-raise after logging

# --- Example Usage ---
if __name__ == '__main__':
    # Setup basic logging for testing this module directly
    log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format) # Use DEBUG level for detailed load info
    logging.getLogger().name = 'file_io_test' # Give logger a name

    # --- Create Dummy Data Files ---
    test_dir = "file_io_test_output"
    os.makedirs(test_dir, exist_ok=True)
    logging.info(f"Test output directory: {os.path.abspath(test_dir)}")

    # File 1: Comma separated, comments, comma decimal
    dummy_csv = os.path.join(test_dir, "dummy_spectrum.csv")
    with open(dummy_csv, "w", encoding='utf-8') as f:
        f.write("# Dummy CSV data\n")
        f.write("# Another comment line\n")
        f.write("Wavelength,Intensity\n") # Header line that should be skipped by comments='#'
        f.write("100,0 , 10\n") # Space after comma, comma decimal
        f.write("101,1, 15 # inline comment\n")
        f.write("102,2, 20 \n") # Trailing space
        f.write("103,3, 18\n")
        f.write("102,2, 99\n") # Duplicate Wavelength
        f.write("104,4, NaN\n") # Invalid intensity
        f.write("BAD, 50\n")   # Invalid wavelength
        f.write("\n")         # Blank line
        f.write("105,5, 12\n")

    # File 2: Tab separated, no comments
    dummy_tsv = os.path.join(test_dir, "dummy_spectrum.txt")
    with open(dummy_tsv, "w", encoding='utf-8') as f:
        f.write("200.0\t50\n")
        f.write("201.5\t55\n")
        f.write("203.0\t48\n")
        f.write("201.5\t999\n") # Duplicate
        f.write("204.5\t52\n")

    # File 3: Space separated (multiple spaces), semicolon comment
    dummy_space = os.path.join(test_dir, "dummy_spectrum.asc")
    with open(dummy_space, "w", encoding='utf-8') as f:
        f.write("; Space separated example\n") # Semicolon comment
        f.write("300.1   1000\n")
        f.write("300.2   1010\n")
        f.write("300.3   1005\n")

    # File 4: Empty file
    dummy_empty = os.path.join(test_dir, "empty.txt")
    open(dummy_empty, 'a').close()

    # File 5: Only comments
    dummy_comments = os.path.join(test_dir, "comments_only.txt")
    with open(dummy_comments, "w", encoding='utf-8') as f:
        f.write("# Line 1\n")
        f.write("# Line 2\n")

    # --- Test Loading ---
    print("\n--- Testing Loading ---")
    loaded_specs = {}
    test_files = {
        'CSV': (dummy_csv, None, '#'),
        'TSV': (dummy_tsv, '\t', None),
        'Space': (dummy_space, None, ';'),
        'Empty': (dummy_empty, None, '#'),
        'CommentsOnly': (dummy_comments, None, '#'),
    }

    for name, (filepath, delim, comm) in test_files.items():
        try:
            print(f"\nLoading: {name} ({os.path.basename(filepath)}) (Delim: {delim or 'Auto'}, Comm: '{comm}')")
            spec = load_spectrum_from_file(filepath, delimiter=delim, comment_char=comm)
            print(f"  -> SUCCESS: Loaded {len(spec)} points.")
            print(f"     Metadata: {spec.metadata}")
            loaded_specs[name] = spec # Store successfully loaded spec
        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"  -> EXPECTED/CAUGHT ERROR for {name}: {type(e).__name__}: {e}")

    # --- Test Saving (using CSV loaded spec if available) ---
    if 'CSV' in loaded_specs:
        spec1 = loaded_specs['CSV']
        print("\n--- Testing Saving (using loaded CSV spec) ---")
        try:
            # Add dummy processed data and peaks for saving tests
            spec1.update_processed(
                 processed_intensity = spec1.raw_intensity * 0.9 + np.random.rand(len(spec1)) * 2,
                 baseline = np.linspace(2, 3, len(spec1))
            )
            spec1.peaks = [
                Peak(index=1, detected_wave=101.1, raw_int=15, proc_int=15.5, fit_results={'center': 101.12, 'amplitude': 14.8, 'sigma': 0.5, 'fwhm': 1.17, 'r_squared': 0.99, 'profile':'Gaussian'}, fit_errors={'center': 0.01, 'amplitude': 0.2, 'sigma': 0.02}),
                Peak(index=3, detected_wave=103.3, raw_int=18, proc_int=18.2, fit_results={'center': 103.31, 'amplitude': 17.9, 'sigma': 0.6, 'fwhm': 1.41, 'r_squared': 0.98, 'profile':'Voigt'}, fit_errors={'center': 0.02, 'amplitude': 0.3, 'sigma': 0.03})
            ]

            # Test saving data
            save_spectrum_path = os.path.join(test_dir, "saved_spectrum.csv")
            save_spectrum_data(spec1, save_spectrum_path, include_processed=True)
            print(f"Saved spectrum data to: {save_spectrum_path}")

            # Test saving peaks
            save_peaks_path = os.path.join(test_dir, "saved_peaks.csv")
            save_peak_list(spec1.peaks, save_peaks_path)
            print(f"Saved peak list to: {save_peaks_path}")

            # Test saving empty peaks
            save_empty_peaks_path = os.path.join(test_dir, "saved_empty_peaks.csv")
            save_peak_list([], save_empty_peaks_path)
            print(f"Saved empty peak list to: {save_empty_peaks_path}")

            # Test saving generic dataframe
            dummy_df = pd.DataFrame({'colA': [1, 2, None], 'colB': [3.14, 6.28, 9.99]})
            save_dummy_df_path = os.path.join(test_dir, "saved_dataframe.csv")
            save_dataframe(dummy_df, save_dummy_df_path, data_description="dummy test data")
            print(f"Saved dummy DataFrame to: {save_dummy_df_path}")

            # Test saving empty dataframe with columns
            save_empty_df_path = os.path.join(test_dir, "saved_empty_dataframe_cols.csv")
            save_dataframe(pd.DataFrame(columns=['A','B']), save_empty_df_path, data_description="empty test data with header")
            print(f"Saved empty DataFrame with header to: {save_empty_df_path}")

            # Test saving None dataframe
            save_none_df_path = os.path.join(test_dir, "saved_none_dataframe.csv")
            save_dataframe(None, save_none_df_path, data_description="None test data")
            print(f"Saved None DataFrame to: {save_none_df_path}")


            print(f"\nSaving tests completed. Check files in '{os.path.abspath(test_dir)}' directory.")

        except Exception as e:
            print(f"\nAn error occurred during saving tests: {e}")
            traceback.print_exc()
    else:
        print("\n--- Skipping Saving Tests (CSV spec failed to load) ---")

    # --- Optional Cleanup ---
    # import shutil
    # try:
    #     user_input = input(f"\nDelete test directory '{test_dir}'? (y/N): ")
    #     if user_input.lower() == 'y':
    #         print(f"Deleting test directory: {test_dir}")
    #         shutil.rmtree(test_dir)
    # except Exception as clean_e:
    #      print(f"Error during cleanup: {clean_e}")