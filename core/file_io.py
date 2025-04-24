# core/file_io.py

"""
Functions for loading and saving spectral data and analysis results.
Handles common text-based spectral file formats and saving various data types to CSV.
"""

import os
import logging
import traceback # Keep for potential detailed error logging if needed later
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any

# Attempt to import core data models from the central location
try:
    # Use relative import assuming data_models is in the same 'core' directory
    from .data_models import Spectrum, Peak, NISTMatch
except ImportError as e_import:
    logging.critical(f"CRITICAL ERROR in file_io.py: Cannot import core data models: {e_import}. File I/O will fail.")
    # Raise the error to prevent the application from potentially running in a broken state
    raise ImportError(f"Core data models failed to import in file_io: {e_import}") from e_import

# --- Constants ---
DEFAULT_SAVE_DELIMITER = ',' # Use comma for CSV saving by default
# Use general format with up to 6 significant digits for saving (balances precision and compactness)
DEFAULT_FLOAT_FORMAT = '%.6g'
DEFAULT_COMMENT_CHAR = '#' # Default character for comments if none specified

# --- Loading Function ---
def load_spectrum_from_file(filepath: str,
                            delimiter: Optional[str] = None,
                            comment_char: Optional[str] = DEFAULT_COMMENT_CHAR,
                            encoding: str = 'utf-8') -> Spectrum:
    """
    Loads spectral data from a text-based file into a Spectrum object.

    Handles common delimiters (tab, comma, space, semicolon) and comment lines.
    Attempts to coerce data to numeric types and handles basic errors.
    Sorts data by wavelength and removes duplicate wavelengths (keeping first).
    Assumes standard '.' decimal separator.

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
        ImportError: Propagated if core Spectrum model couldn't be imported initially.
        Exception: For other unexpected errors during loading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    logging.info(f"Attempting load: '{os.path.basename(filepath)}' (Delimiter: {delimiter or 'Auto'}, Comment: '{comment_char}')")

    used_comment = comment_char if comment_char else None
    guessed_delimiter = delimiter

    try:
        # --- Delimiter Guessing (if not provided) ---
        if delimiter is None:
            logging.debug(f"No delimiter provided, attempting to guess for {os.path.basename(filepath)}.")
            try:
                # Read first few non-empty, non-comment lines for guessing
                with open(filepath, 'r', encoding=encoding) as f:
                    lines_to_check = []
                    for _ in range(10): # Check up to 10 lines
                        line = f.readline()
                        if not line: break # End of file
                        line_strip = line.strip()
                        if line_strip and (not used_comment or not line_strip.startswith(used_comment)):
                            lines_to_check.append(line_strip)
                        if len(lines_to_check) >= 5: break # Usually enough

                if not lines_to_check:
                    raise ValueError("File appears empty or contains only comment lines.")

                first_data_line = lines_to_check[0]
                logging.debug(f"First data line for guessing: '{first_data_line}'")

                # Simple guessing logic (can be improved with csv.Sniffer if needed)
                if '\t' in first_data_line: guessed_delimiter = '\t'
                elif ',' in first_data_line: guessed_delimiter = ','
                elif ';' in first_data_line: guessed_delimiter = ';'
                elif ' ' in first_data_line:
                    # Check if space-separated looks numeric (simple check)
                    parts = first_data_line.split()
                    try:
                        if len(parts) >= 2 and all('.' in p or p.isdigit() or ('e' in p.lower()) or ('-' in p) for p in parts[:2]):
                            guessed_delimiter = r'\s+' # Regex for one or more spaces
                        else:
                            raise ValueError("Space separated but not clearly numeric pairs")
                    except ValueError:
                        logging.debug("Space found, but parts don't look numeric. Falling back to Tab.")
                        guessed_delimiter = '\t' # Fallback if space doesn't look right
                else:
                     # Default fallback if no common delimiter found
                    guessed_delimiter = '\t'
                    logging.warning(f"Could not guess delimiter for {os.path.basename(filepath)}, defaulting to Tab.")

                logging.info(f"Guessed delimiter: {repr(guessed_delimiter)} for {os.path.basename(filepath)}")
            except Exception as guess_e:
                logging.warning(f"Delimiter guessing failed ({guess_e}), defaulting to Tab ('\\t').")
                guessed_delimiter = '\t'

        # --- Read using Pandas ---
        logging.debug(f"Reading with Pandas: sep={repr(guessed_delimiter)}, comment={repr(used_comment)}")
        try:
            # Read as string first to handle potential mixed types or locale issues manually if needed
            df = pd.read_csv(
                filepath,
                sep=guessed_delimiter,
                header=None,
                names=["Wavelength", "Intensity"], # Assume 2 columns
                comment=used_comment,
                encoding=encoding,
                skipinitialspace=True,
                skip_blank_lines=True,
                on_bad_lines='warn', # Log problematic lines but try to continue
                dtype=str,           # Read as string initially
                engine='python'      # More flexible engine, handles regex delimiters better
            )
        except pd.errors.EmptyDataError:
            raise ValueError("File is empty or contains only comments/blank lines.") from None
        except pd.errors.ParserError as pe:
            logging.error(f"Pandas parsing error: {pe}. Check delimiter ({repr(guessed_delimiter)}), comments ('{used_comment}'), and file structure.")
            raise ValueError(f"Failed to parse file structure. Check delimiter/comments. Error: {pe}") from pe
        except Exception as read_e:
            logging.error(f"Error reading file with Pandas: {read_e}", exc_info=True)
            raise IOError(f"Could not read file {os.path.basename(filepath)}.") from read_e

        if df.empty:
            raise ValueError("No data could be parsed (potentially only comments/blank/bad lines).")

        original_rows = len(df)
        logging.debug(f"Read {original_rows} lines initially.")

        # --- Data Cleaning and Conversion ---
        # WARNING: Removed comma->period replacement. Assumes standard '.' decimal separator.
        # This is safer for different locales or comma-delimited files.
        # If comma decimals are expected, handle this based on config or user input.
        # logging.debug("Assuming '.' decimal separator. Comma replacement removed.")
        df['Wavelength'] = pd.to_numeric(df['Wavelength'].astype(str), errors='coerce')
        df['Intensity'] = pd.to_numeric(df['Intensity'].astype(str), errors='coerce')

        # Drop rows where conversion failed
        initial_nan_count = df.isnull().any(axis=1).sum()
        rows_before_dropna = len(df)
        df.dropna(subset=['Wavelength', 'Intensity'], inplace=True)
        rows_after_dropna = len(df)

        if initial_nan_count > 0:
             logging.warning(f"Removed {rows_before_dropna - rows_after_dropna} rows with non-numeric values.")
        if df.empty:
            raise ValueError(f"No valid numeric Wavelength/Intensity pairs found after cleaning {original_rows} initial lines. Check file content and format.")

        # --- Extract NumPy Arrays and Check Finite ---
        wavelengths = df['Wavelength'].to_numpy(dtype=float)
        raw_intensity = df['Intensity'].to_numpy(dtype=float)

        # Filter out non-finite values (Inf/NaN) that might remain after coerce
        finite_mask = np.isfinite(wavelengths) & np.isfinite(raw_intensity)
        if not np.all(finite_mask):
            num_nonfinite = (~finite_mask).sum()
            logging.warning(f"Found and removing {num_nonfinite} non-finite values (Inf/NaN) after conversion.")
            wavelengths = wavelengths[finite_mask]
            raw_intensity = raw_intensity[finite_mask]

        final_finite_rows = len(wavelengths)
        if final_finite_rows < 2: # Require at least 2 points for meaningful spectrum
            raise ValueError(f"Spectrum must contain at least 2 valid finite data points. Found {final_finite_rows} after all cleaning.")

        # --- Sorting and Deduplication ---
        logging.debug("Sorting data by wavelength...")
        sort_indices = np.argsort(wavelengths)
        if not np.array_equal(sort_indices, np.arange(final_finite_rows)):
            logging.info(f"Wavelength data in {os.path.basename(filepath)} was not sorted. Sorting now.")
            wavelengths = wavelengths[sort_indices]
            raw_intensity = raw_intensity[sort_indices]

        logging.debug("Checking for duplicate wavelengths...")
        unique_wavelengths, unique_indices = np.unique(wavelengths, return_index=True)
        final_unique_rows = len(unique_wavelengths)
        if final_unique_rows < final_finite_rows:
            num_dupes = final_finite_rows - final_unique_rows
            logging.warning(f"Found and removing {num_dupes} duplicate wavelength entries (keeping first occurrence).")
            wavelengths = wavelengths[unique_indices]
            raw_intensity = raw_intensity[unique_indices]

        if final_unique_rows < 2: # Check again after deduplication
            raise ValueError("Spectrum has less than 2 unique valid points after deduplication.")

        # --- Create Spectrum Object ---
        metadata = {
            "original_filename": os.path.basename(filepath),
            "load_timestamp": pd.Timestamp.now().isoformat(),
            "initial_rows_read": original_rows,
            "rows_after_conversion_drop": rows_after_dropna,
            "rows_after_finite_filter": final_finite_rows,
            "final_points": final_unique_rows,
            "guessed_delimiter": repr(guessed_delimiter), # Store repr for clarity (\t vs ' ')
            "used_comment_char": repr(used_comment)
        }
        spectrum = Spectrum(
            wavelengths=wavelengths,
            raw_intensity=raw_intensity,
            metadata=metadata,
            source_filepath=filepath # Store full path
        )
        logging.info(f"Successfully loaded: {spectrum}")
        return spectrum

    # --- More Specific Exception Handling ---
    except FileNotFoundError:
        logging.error(f"File not found during load: {filepath}")
        raise # Re-raise specific error
    except ValueError as ve:
        # Catch ValueErrors raised internally (e.g., parsing, data validation)
        logging.error(f"Invalid format or data error loading {os.path.basename(filepath)}: {ve}")
        # Raise a new ValueError with combined info
        raise ValueError(f"Invalid format or data in '{os.path.basename(filepath)}': {ve}") from ve
    except ImportError:
        raise # Re-raise the import error if models weren't available initially
    except Exception as e:
        # Catch other potential errors (IOError, MemoryError, etc.)
        logging.error(f"Unexpected error loading {os.path.basename(filepath)}: {e}", exc_info=True)
        # Raise a generic Exception to indicate load failure
        raise Exception(f"Failed to load '{os.path.basename(filepath)}': {e}") from e


# --- Saving Functions ---

def save_spectrum_data(spectrum: Spectrum, filepath: str, include_processed: bool = True,
                         delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """
    Saves spectrum data (raw, optionally processed and baseline) to a CSV file.

    Args:
        spectrum (Spectrum): The Spectrum object containing the data.
        filepath (str): The full path for the output file.
        include_processed (bool): If True, include processed_intensity and baseline if available.
        delimiter (str): Delimiter for the CSV file.
        float_format (str): Format string for floating-point numbers (e.g., '%.6g').
    """
    if not isinstance(spectrum, Spectrum) or len(spectrum) == 0:
        logging.warning(f"Attempted to save empty or invalid spectrum object ({type(spectrum)}). Skipping save to {filepath}.")
        return
    if not filepath:
        logging.error("Cannot save spectrum: Filepath is empty.")
        raise ValueError("Filepath cannot be empty for saving spectrum data.")

    logging.info(f"Saving spectrum data ({len(spectrum)} points) to: {filepath}")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists

        data_dict: Dict[str, np.ndarray] = {
            "Wavelength_nm": spectrum.wavelengths,
            "Raw_Intensity": spectrum.raw_intensity
        }
        columns = ["Wavelength_nm", "Raw_Intensity"]

        if include_processed:
            if spectrum.processed_intensity is not None:
                if spectrum.processed_intensity.shape == spectrum.wavelengths.shape:
                    data_dict["Processed_Intensity"] = spectrum.processed_intensity
                    columns.append("Processed_Intensity")
                    # Only include baseline if processed is included and baseline exists/matches shape
                    if spectrum.baseline is not None:
                         if spectrum.baseline.shape == spectrum.wavelengths.shape:
                              data_dict["Baseline"] = spectrum.baseline
                              columns.append("Baseline")
                              logging.debug("Including baseline in saved spectrum data.")
                         else:
                              logging.warning("Baseline length mismatch, baseline NOT saved.")
                    logging.debug("Including processed intensity in saved spectrum data.")
                else:
                    logging.warning("Processed intensity shape mismatch. Processed data NOT saved.")
            elif spectrum.baseline is not None:
                 logging.warning("Baseline exists but processed intensity is missing. Baseline NOT saved.")


        df_to_save = pd.DataFrame(data_dict)[columns] # Ensure correct column order
        df_to_save.to_csv(
            filepath,
            sep=delimiter,
            header=True,
            index=False,
            float_format=float_format,
            encoding='utf-8',
            na_rep='NaN', # Representation for Not a Number
            lineterminator='\n' # Use standard newline
        )
        logging.info(f"Spectrum data saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save spectrum data to {filepath}: {e}", exc_info=True)
        raise # Re-raise exception after logging


def save_peak_list(peaks: List[Peak], filepath: str,
                   delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """
    Saves the detected/fitted peak list to a CSV file.

    Ensures all standard columns are present, using NaN for missing values.

    Args:
        peaks (List[Peak]): The list of Peak objects to save.
        filepath (str): The full path for the output file.
        delimiter (str): Delimiter for the CSV file.
        float_format (str): Format string for floating-point numbers.
    """
    if not filepath:
        logging.error("Cannot save peak list: Filepath is empty.")
        raise ValueError("Filepath cannot be empty for saving peak list.")

    logging.info(f"Attempting to save peak list ({len(peaks)} peaks) to: {filepath}")

    # Define the standard output columns, based on Peak.to_dataframe_row and FitResult.get_param_dict
    output_columns = [
        "Peak Index", "Detected Wavelength (nm)", "Raw Intensity", "Processed Intensity",
        "Fit Profile", "Fitted Center (nm)", "Fitted Amplitude",
        "Fitted Sigma/Gamma (nm)", # Clarified name
        "Fitted FWHM (nm)", "Fitted Area (a.u.)", "Fit Mixing (eta)",
        "Fit R^2", "Fit AIC", "Fit BIC",
        "Fit Amp Error", "Fit Cen Error", "Fit Wid Error",
        "Fit Eta Error", "Fit Area Error"
        # Add optional NIST summary columns here if Peak.to_dataframe_row includes them
    ]

    os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure directory exists

    if not peaks:
        logging.warning("Peak list is empty. Saving CSV file with header only.")
        try:
            pd.DataFrame(columns=output_columns).to_csv(
                filepath, sep=delimiter, index=False, encoding='utf-8', lineterminator='\n'
            )
        except Exception as e:
            logging.error(f"Failed to save empty peak list header: {e}", exc_info=True)
            raise
        return

    try:
        # Convert list of Peak objects to list of dictionaries
        peak_data_list = []
        for i, peak in enumerate(peaks):
             if isinstance(peak, Peak) and hasattr(peak, 'to_dataframe_row'):
                 try:
                     row_data = peak.to_dataframe_row()
                     if isinstance(row_data, dict):
                         peak_data_list.append(row_data)
                     else:
                         logging.warning(f"Peak {i} 'to_dataframe_row' did not return a dictionary.")
                 except Exception as row_e:
                     logging.error(f"Error converting peak {i} (Index: {getattr(peak, 'index', '?')}) to row: {row_e}.", exc_info=False)
             else:
                 logging.error(f"Item at index {i} in peaks list is not a valid Peak object.")

        if not peak_data_list:
            logging.error("No valid peak data could be extracted for saving.")
            # Save header only even if conversion failed for all
            pd.DataFrame(columns=output_columns).to_csv(filepath, sep=delimiter, index=False, encoding='utf-8', lineterminator='\n')
            return

        # Create DataFrame from list of dictionaries
        peak_df = pd.DataFrame(peak_data_list)
        logging.debug(f"Created DataFrame with {len(peak_df)} rows from peak list.")

        # Ensure all standard columns exist and are in the correct order
        # Reindex fills missing columns with NaN automatically
        peak_df_to_save = peak_df.reindex(columns=output_columns)

        # Check if any expected columns were *completely* missing from the generated data
        missing_cols = set(output_columns) - set(peak_df.columns)
        if missing_cols:
             logging.warning(f"The following columns were expected but not generated by any peak: {missing_cols}. They will contain only NaN values.")

        # Save the final DataFrame
        peak_df_to_save.to_csv(
            filepath, sep=delimiter, header=True, index=False,
            float_format=float_format, encoding='utf-8', na_rep='NaN', lineterminator='\n'
        )
        logging.info(f"Peak list ({len(peak_df_to_save)} peaks) saved successfully to {filepath}")

    except Exception as e:
        logging.error(f"Failed to save peak list to {filepath}: {e}", exc_info=True)
        raise


def save_nist_matches(matches_df: Optional[pd.DataFrame], filepath: str,
                      delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT):
    """ Saves the NIST match results DataFrame (if valid) to a CSV file. """
    if not isinstance(matches_df, pd.DataFrame) and matches_df is not None:
        logging.error("Invalid data type passed to save_nist_matches (must be DataFrame or None).")
        raise TypeError("Data must be a pandas DataFrame or None.")
    save_dataframe(matches_df, filepath, delimiter, float_format, "NIST matches")


def save_dataframe(df: Optional[pd.DataFrame], filepath: str,
                   delimiter: str = DEFAULT_SAVE_DELIMITER, float_format: str = DEFAULT_FLOAT_FORMAT,
                   data_description: str = "data"):
    """ Generic function to save any pandas DataFrame to CSV with error handling. """
    desc_cap = data_description.capitalize()
    is_valid_df = isinstance(df, pd.DataFrame)
    shape_info = f"({df.shape[0]}x{df.shape[1]})" if is_valid_df and not df.empty else "(None or Empty)"
    logging.info(f"Attempting to save {data_description} {shape_info} to: {filepath}")

    if not filepath:
        logging.error(f"Cannot save {data_description}: filepath is empty.")
        raise ValueError("Invalid filepath for saving DataFrame.")

    try:
        # Ensure directory exists
        dir_name = os.path.dirname(filepath)
        if dir_name: # Only create if path has a directory component
             os.makedirs(dir_name, exist_ok=True)

        # Handle None or empty DataFrame case by saving header only
        if not is_valid_df or df.empty:
            log_msg = f"{desc_cap} DataFrame is None." if df is None else f"{desc_cap} DataFrame is empty."
            logging.warning(f"{log_msg} Saving file with header only.")
            # Get columns from df if possible, otherwise empty list
            cols_to_write = df.columns if is_valid_df and not df.empty else []
            pd.DataFrame(columns=cols_to_write).to_csv(
                filepath, sep=delimiter, index=False, encoding='utf-8', lineterminator='\n'
            )
            logging.info(f"Saved empty {data_description} file (header only) to {filepath}.")
            return

        # Save the valid, non-empty DataFrame
        df.to_csv(
            filepath,
            sep=delimiter,
            header=True,
            index=False,
            float_format=float_format,
            encoding='utf-8',
            na_rep='NaN',
            lineterminator='\n'
        )
        logging.info(f"{desc_cap} saved successfully ({df.shape[0]} rows) to {filepath}")

    except Exception as e:
        logging.error(f"Failed to save {data_description} to {filepath}: {e}", exc_info=True)
        raise # Re-raise exception after logging

# --- Example Usage Block (Optional) ---
# if __name__ == '__main__':
#     log_format = '%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s'
#     logging.basicConfig(level=logging.DEBUG, format=log_format)
#     logging.getLogger().name = 'file_io_test'
#     test_dir = "file_io_test_output"
#     os.makedirs(test_dir, exist_ok=True)
#     logging.info(f"Test output directory: {os.path.abspath(test_dir)}")
#     # Add test cases here using dummy Spectrum/Peak objects if needed
#     # Ensure core.data_models can be imported for tests to run