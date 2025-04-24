# -*- coding: utf-8 -*-
"""
Handles saving and loading the application state (session).

This allows users to save their work (loaded spectrum reference, processing settings,
analysis results like peaks, fits, plasma parameters) and resume later.

Uses JSON for storing serializable state information. Large data arrays (like
raw spectral data) are NOT stored directly; instead, file paths are saved,
and data is expected to be reloaded from the original source. Derived data
like DataFrames and analysis results (Peaks, Fits, Matches) are converted to
a serializable dictionary format using custom methods (e.g., .to_dict()).
"""

import os
import json
import logging
import traceback # Keep for debugging if needed
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import issues with MainWindow for type hints
if TYPE_CHECKING:
    from ui.main_window import MainWindow

# Import core data models (needed for type checking and .to_dict() assumptions)
try:
    # Use relative import assuming standard structure
    from .data_models import Spectrum, Peak, FitResult, NISTMatch
except ImportError as e_import:
    logging.critical(f"CRITICAL ERROR in session_manager.py: Cannot import core data models: {e_import}.")
    raise ImportError(f"Core data models failed to import in session_manager: {e_import}") from e_import

# --- Constants ---
SESSION_FILE_EXTENSION = ".lcfses" # LIBS Cosmic Forge Session
# Increment version if the structure/serialization method changes significantly
SESSION_VERSION = "1.1" # Version 1.1 reflects using to_dict for peaks/matches/fits

# --- Custom JSON Encoder for NumPy/Pandas types ---
class DataEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle common non-serializable types like NumPy scalars,
    Pandas Timestamps, and explicitly prevent direct serialization of NumPy arrays.
    """
    def default(self, obj):
        # NumPy Scalars (integers, floats, booleans)
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            # Handle NaN/Inf appropriately for JSON
            if np.isnan(obj):
                return None # Represent NaN as JSON null (standard)
            elif np.isinf(obj):
                # Represent Inf as strings "Infinity" / "-Infinity"
                # The loading code MUST handle converting these back to np.inf
                return str(obj)
            else:
                return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        # NumPy Arrays (explicitly PREVENT serialization)
        elif isinstance(obj, np.ndarray):
            # Use DEBUG level as this is expected behavior, not necessarily a warning
            logging.debug("Attempted to serialize NumPy array directly in session. Skipping.")
            return "<NumPy Array (not saved)>" # Placeholder string indicates skipped data
        # Pandas Timestamps (convert to standard ISO format string)
        elif isinstance(obj, pd.Timestamp):
             return obj.isoformat()
        # Other specific types if needed (e.g., np.void for structured arrays if used)
        elif isinstance(obj, (np.void)):
            logging.debug(f"Serializing np.void object as None.")
            return None
        # Let the base class default method raise the TypeError for other unhandled types
        try:
            return super().default(obj) # Use super() instead of direct call
        except TypeError:
            # Log error with type and representation for better debugging
            logging.error(f"Object of type {type(obj)} is not JSON serializable. Value: {obj!r}")
            raise # Re-raise the error after logging


class SessionManager:
    """Manages saving and loading application session state."""

    SESSION_FILE_EXTENSION = SESSION_FILE_EXTENSION # Expose for use elsewhere

    def __init__(self, main_window: 'MainWindow'):
        """
        Initializes the SessionManager.

        Args:
            main_window: A reference to the main application window instance (`MainWindow`).
                         Needed to access current state and UI elements.
        """
        if not hasattr(main_window, 'config') or not hasattr(main_window, '_is_busy'):
             # Basic check for essential main_window attributes
             raise TypeError("SessionManager requires a MainWindow instance with 'config' and '_is_busy' attributes.")
        self.main_window: 'MainWindow' = main_window


    def _get_panel_state(self, panel_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Helper to safely retrieve settings from a UI panel widget associated with the main window.

        Checks common panel naming conventions and state retrieval method names ('get_settings', 'get_state').

        Args:
            panel_identifier: The key used to identify the dock widget or panel attribute
                              (e.g., 'processing', 'nist_search').

        Returns:
            A dictionary containing the panel's state, or None if retrieval fails.
        """
        panel_widget = None
        mw = self.main_window
        try:
            # Try accessing panel via typical attribute names
            potential_attrs = [f"{panel_identifier}_panel", f"{panel_identifier}_view"]
            for attr in potential_attrs:
                if hasattr(mw, attr):
                    panel_widget = getattr(mw, attr, None)
                    if panel_widget: break

            # If not found as attribute, check if it's a dock widget
            if panel_widget is None and hasattr(mw, 'docks') and panel_identifier in mw.docks:
                 dock = mw.docks.get(panel_identifier)
                 if dock and hasattr(dock, 'widget'):
                      panel_widget = dock.widget()

            # If panel widget found, try calling state retrieval methods
            if panel_widget:
                 if hasattr(panel_widget, 'get_settings') and callable(panel_widget.get_settings):
                     return panel_widget.get_settings()
                 elif hasattr(panel_widget, 'get_state') and callable(panel_widget.get_state):
                      return panel_widget.get_state()
                 else:
                      logging.debug(f"Panel/View '{panel_identifier}' found, but has no standard state retrieval method ('get_settings' or 'get_state').")
            else:
                 # It's okay if some panels don't exist or don't need saving state
                 logging.debug(f"Panel/View/Dock for '{panel_identifier}' not found for state retrieval.")

        except Exception as e:
            # Log specific panel error but allow session gathering to continue
            logging.warning(f"Could not get state for panel '{panel_identifier}': {e}", exc_info=True)

        return None # Return None if no state found or error occurred


    def _serialize_dataframe(self, df: Optional[pd.DataFrame]) -> Optional[List[Dict[str, Any]]]:
        """
        Converts a Pandas DataFrame to a JSON-serializable list of records (dictionaries).

        Args:
            df: The DataFrame to serialize.

        Returns:
            A list of dictionaries (orient='records'), an empty list for an empty DataFrame,
            or None if input is not a DataFrame or serialization fails.
        """
        if df is None:
            return None # Return None if input is None
        if isinstance(df, pd.DataFrame):
            if not df.empty:
                 try:
                     # 'records' format [{col->val}, {col->val}, ...] is generally easy to reconstruct
                     # using pd.DataFrame.from_records(loaded_list)
                     # Handle potential non-finite floats before serialization if needed?
                     # df_serializable = df.replace([np.inf, -np.inf], [None, None]) # Example
                     # return df_serializable.to_dict(orient='records')
                     return df.to_dict(orient='records')
                 except Exception as e:
                     logging.error(f"Failed to serialize DataFrame to dict (orient='records'): {e}", exc_info=True)
                     return None # Indicate serialization failure
            else:
                 return [] # Return empty list for empty DataFrame
        else:
             logging.warning(f"Attempted to serialize non-DataFrame object: {type(df)}. Returning None.")
             return None


    def gather_session_data(self) -> Dict[str, Any]:
        """
        Collects the current application state into a serializable dictionary.

        This includes window state, paths, settings, and analysis results serialized
        using their `.to_dict()` methods.

        Returns:
            A dictionary representing the application state ready for JSON serialization.
            Returns an incomplete dictionary if critical parts fail.
        """
        state: Dict[str, Any] = {"session_version": SESSION_VERSION}
        mw = self.main_window # Shorthand for main window reference

        # --- Main Window State ---
        try:
            state['window_geometry'] = mw.saveGeometry().toBase64().data().decode('ascii')
            state['window_state'] = mw.saveState().toBase64().data().decode('ascii')
            state['current_theme'] = mw.theme_manager.current_theme_name if hasattr(mw, 'theme_manager') else None
            state['last_load_dir'] = getattr(mw, '_last_load_dir', os.path.expanduser("~"))
            state['last_save_dir'] = getattr(mw, '_last_save_dir', os.path.expanduser("~"))
        except Exception as e:
            logging.error(f"Failed to gather main window state/paths/theme: {e}", exc_info=True)
            # Decide if this is critical - perhaps return {} or raise? Continue for now.

        # --- Loaded Data References (Paths Only) ---
        try:
            if mw.current_spectrum and hasattr(mw.current_spectrum, 'source_filepath') and mw.current_spectrum.source_filepath:
                state['current_spectrum_path'] = mw.current_spectrum.source_filepath
                # Store loader params if available in metadata
                if hasattr(mw.current_spectrum, 'metadata') and isinstance(mw.current_spectrum.metadata, dict):
                    state['current_spectrum_delimiter'] = mw.current_spectrum.metadata.get('guessed_delimiter') # Store repr
                    state['current_spectrum_comment'] = mw.current_spectrum.metadata.get('used_comment_char') # Store repr
            else:
                state['current_spectrum_path'] = None

            if hasattr(mw, 'multi_spectra') and isinstance(mw.multi_spectra, list):
                state['multi_spectra_paths'] = [
                    s.source_filepath for s in mw.multi_spectra
                    if s and hasattr(s, 'source_filepath') and s.source_filepath
                ]
            else:
                state['multi_spectra_paths'] = []
        except Exception as e:
            logging.error(f"Failed to gather loaded spectra paths: {e}", exc_info=True)


        # --- UI Panel States ---
        panel_keys = ['processing', 'detection', 'fitting', 'nist_search',
                      'boltzmann', 'cflibs', 'ml_analysis', 'peak_list'] # Use consistent keys
        for key in panel_keys:
             try:
                  panel_state = self._get_panel_state(key)
                  # Only store if state is not None (avoid null entries for missing panels)
                  if panel_state is not None:
                      state[f"{key}_settings"] = panel_state
             except Exception as e: # Catch errors during state retrieval for a single panel
                  logging.error(f"Failed to get state for panel '{key}': {e}", exc_info=True)


        # --- Analysis Results (Using .to_dict() methods) ---
        try:
            # Serialize Peak data using Peak.to_dict()
            if hasattr(mw, 'detected_peaks') and isinstance(mw.detected_peaks, list):
                serializable_peaks = []
                for p in mw.detected_peaks:
                    if hasattr(p, 'to_dict') and callable(p.to_dict):
                        serializable_peaks.append(p.to_dict())
                    else:
                        logging.warning(f"Peak object {p!r} missing to_dict method. Skipping.")
                state['detected_peaks'] = serializable_peaks
            else:
                state['detected_peaks'] = []
        except Exception as e:
             logging.error(f"Failed to serialize detected peaks: {e}", exc_info=True)
             state['detected_peaks'] = [] # Ensure key exists but is empty on error

        try:
            # Serialize NIST matches using NISTMatch.to_dict()
            if hasattr(mw, 'nist_matches') and isinstance(mw.nist_matches, list):
                serializable_matches = []
                for m in mw.nist_matches:
                     if hasattr(m, 'to_dict') and callable(m.to_dict):
                          serializable_matches.append(m.to_dict())
                     else:
                          logging.warning(f"NISTMatch object {m!r} missing to_dict method. Skipping.")
                state['nist_matches'] = serializable_matches
            else:
                state['nist_matches'] = []
        except Exception as e:
             logging.error(f"Failed to serialize NIST matches: {e}", exc_info=True)
             state['nist_matches'] = []

        try:
            # Serialize DataFrames derived from analysis using helper
            state['boltzmann_plot_data'] = self._serialize_dataframe(getattr(mw, 'boltzmann_plot_data', None))
            state['cf_libs_concentrations'] = self._serialize_dataframe(getattr(mw, 'cf_libs_concentrations', None))
        except Exception as e:
             logging.error(f"Failed to serialize analysis DataFrames: {e}", exc_info=True)
             state['boltzmann_plot_data'] = None # Set to None on error
             state['cf_libs_concentrations'] = None

        try:
            # --- Plasma Parameters (should be directly serializable) ---
            state['plasma_temp_k'] = getattr(mw, 'plasma_temp_k', None)
            state['electron_density_cm3'] = getattr(mw, 'electron_density_cm3', None)
        except Exception as e:
             logging.error(f"Failed to gather plasma parameters: {e}", exc_info=True)


        # --- Plot State (Generally avoid saving this) ---
        # Saving plot limits can be fragile. Best practice is usually to replot from data.
        # If absolutely needed, uncomment and test thoroughly:
        # try:
        #     if mw.plot_widget and mw.plot_widget.ax and mw.plot_widget.ax.has_data():
        #         state['plot_xlim'] = mw.plot_widget.ax.get_xlim()
        #         state['plot_ylim'] = mw.plot_widget.ax.get_ylim()
        # except Exception as e:
        #      logging.debug(f"Could not get plot limits for session save: {e}")

        logging.info("Session data gathering complete.")
        logging.debug(f"Session State Keys Gathered: {list(state.keys())}")
        return state


    def save_session(self, filepath: str) -> bool:
        """
        Saves the current application state to a JSON file using the custom encoder.

        Args:
            filepath: The full path to the file where the session will be saved.

        Returns:
            True if saving was successful, False otherwise.
        """
        mw = self.main_window # Shorthand

        # Prevent saving during critical operations (check main window's busy flag FIRST)
        if getattr(mw, '_is_busy', False): # Use getattr for safety
             logging.warning("Save session request ignored: Application is busy.")
             if hasattr(mw, 'update_status') and callable(mw.update_status):
                  mw.update_status("Cannot save session: Busy.", 3000)
             return False

        logging.info(f"Attempting to save session to: {filepath}")

        # 1. Gather the current state
        try:
            session_data = self.gather_session_data()
            if not session_data: # Check if gathering itself failed critically
                 raise RuntimeError("Session data gathering returned empty or None.")
        except Exception as e:
             logging.error(f"Failed to gather session data for saving: {e}", exc_info=True)
             if hasattr(mw, 'update_status'): mw.update_status("Failed to gather session data.", 5000)
             return False

        # 2. Serialize and write to file
        try:
            # Ensure the target directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path: # Only create if not saving to root (e.g., '/')
                 os.makedirs(dir_path, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                # Use the custom encoder to handle NumPy types etc.
                json.dump(session_data, f, indent=4, cls=DataEncoder, ensure_ascii=False)

            logging.info(f"Session saved successfully to {filepath}")
            if hasattr(mw, 'update_status'): mw.update_status(f"Session saved: {os.path.basename(filepath)}", 5000)
            return True

        except TypeError as e_serial:
             # This catches errors during JSON serialization (e.g., unhandled types from DataEncoder)
             logging.error(f"Serialization error saving session to {filepath}: {e_serial}. Check custom encoder and data types.", exc_info=True)
             if hasattr(mw, 'update_status'): mw.update_status(f"Save Error: Data type issue ({e_serial}).", 5000)
             return False
        except IOError as e_io:
             # Catches file write errors (permissions, disk full, etc.)
             logging.error(f"File I/O error saving session to {filepath}: {e_io}", exc_info=True)
             if hasattr(mw, 'update_status'): mw.update_status(f"Save Error: Could not write file ({e_io}).", 5000)
             return False
        except Exception as e_other:
            # Catch other unexpected errors during saving
            logging.error(f"Unexpected error saving session state to {filepath}: {e_other}", exc_info=True)
            if hasattr(mw, 'update_status'): mw.update_status(f"Save Error: Unexpected error ({e_other}).", 5000)
            return False


    def load_session_data(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Loads session state from a JSON file into a dictionary.

        Performs validation for file existence, JSON format, and session version.
        Does **not** apply the state to the application; that is the responsibility
        of the caller (typically MainWindow._apply_loaded_session_state).

        Args:
            filepath: The path to the session file (.lcfses).

        Returns:
            A dictionary containing the loaded session state, or None if loading fails.

        Raises:
            FileNotFoundError: If the specified filepath does not exist.
            ValueError: If the file is not valid JSON, missing essential keys, or fails validation.
            IOError: For other file reading issues.
        """
        logging.info(f"Attempting to load session data from: {filepath}")
        if not os.path.isfile(filepath):
             logging.error(f"Session file not found: {filepath}")
             raise FileNotFoundError(f"Session file not found: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f) # Standard JSON load

            if not isinstance(state, dict) or not state:
                # Check if load resulted in non-dict or empty dict
                raise ValueError("Session file is empty or does not contain a valid JSON object.")

            # --- Version Check ---
            file_version = state.get("session_version")
            if file_version is None:
                logging.warning(f"Session file '{os.path.basename(filepath)}' has no version information. Loading may be incomplete or fail.")
                # Allow loading older unversioned files for now, but warn heavily.
                # Could raise ValueError here for stricter handling.
            elif file_version != SESSION_VERSION:
                 logging.warning(f"Session file version mismatch (File: {file_version}, App requires: {SESSION_VERSION}). "
                                 "Loading may be incomplete or results may be unexpected.")
                 # Future: Implement version migration logic if needed. For now, allow loading.

            # --- Basic Validation (Check for essential keys needed for minimal restore) ---
            # These keys are fundamental for restoring the basic application context.
            # Add more keys here if they become absolutely critical for core function.
            essential_keys = ['window_geometry', 'window_state', 'current_theme', 'current_spectrum_path']
            missing_keys = [key for key in essential_keys if key not in state]
            if missing_keys:
                 raise ValueError(f"Session file is missing essential data key(s): {', '.join(missing_keys)}")

            logging.info(f"Session data loaded successfully from {os.path.basename(filepath)} (Version: {file_version}).")
            return state

        except json.JSONDecodeError as e_json:
            logging.error(f"Error decoding session file (invalid JSON) {filepath}: {e_json}", exc_info=True)
            # Raise ValueError for MainWindow to catch and inform user about corruption
            raise ValueError(f"Could not parse session file (invalid JSON): {e_json}") from e_json
        except ValueError as e_val: # Catch our own ValueErrors from empty/missing keys
             logging.error(f"Validation error in session file {filepath}: {e_val}", exc_info=False) # Log details
             raise # Re-raise the specific ValueError
        except IOError as e_io:
             logging.error(f"Failed to read session state data from {filepath}: {e_io}", exc_info=True)
             raise # Re-raise IOError
        except Exception as e_other:
            # Catch other potential errors during file reading or initial processing
            logging.error(f"Failed to load session state data from {filepath}: {e_other}", exc_info=True)
            # Raise a more generic Exception indicating a problem reading the file
            raise IOError(f"Failed to load session file '{os.path.basename(filepath)}': An unexpected error occurred.") from e_other