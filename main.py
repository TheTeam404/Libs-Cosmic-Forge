# -*- coding: utf-8 -*-
"""
Main Entry Point for the LIBS Cosmic Forge Application.

Initializes the application environment (paths, logging, configuration),
sets up global exception handling, and launches the main user interface window.
"""

import sys
import os
import traceback
import logging
import copy # For deep copy in config loading

# --- Third-party Imports ---
# Attempt critical imports first and provide informative errors
_QT_AVAILABLE = False
_YAML_AVAILABLE = False
try:
    import yaml
    _YAML_AVAILABLE = True
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import Qt, QCoreApplication, QSettings
    _QT_AVAILABLE = True
except ImportError as e:
    # Minimal error message if basic Qt/YAML is missing
    print(f"[CRITICAL ERROR] Missing essential libraries: {e}")
    print("This application requires Python 3.10+ and several libraries.")
    print("Please ensure all dependencies are installed, typically via:")
    print("  pip install -r requirements.txt")
    # Attempt to show a simple message box if QApplication exists
    try:
        if _QT_AVAILABLE:
            app_temp = QApplication([]) # Attempt to create a temporary app
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.setWindowTitle("Startup Error - Missing Library")
            error_box.setText(f"Missing essential library: {e}.\nPlease install requirements (see console).")
            error_box.exec()
    except Exception as msg_e:
        print(f"[ERROR] Could not display GUI error message: {msg_e}")
    sys.exit(1)

# --- Application Metadata ---
APP_VERSION = "1.0.1" # Updated version reflecting fixes
ORG_NAME = "CosmicForgeDev"
APP_NAME = "LIBSForge"

# --- Set Application Attributes EARLY ---
# Must be done BEFORE QSettings() is instantiated for the first time
# to ensure settings are stored in the correct location.
QCoreApplication.setOrganizationName(ORG_NAME)
QCoreApplication.setApplicationName(APP_NAME)
QCoreApplication.setApplicationVersion(APP_VERSION)
# Required for High DPI display scaling support
try:
     QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
     QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
except AttributeError:
     print("[WARNING] Could not set HighDpi attributes (likely older PyQt version). Scaling may be incorrect.")


# --- Early Path Setup & Logging Configuration ---
_project_root_early = None
try:
    # Attempt to import helper to find root path needed for logging/config
    # This assumes utils/helpers.py is importable relative to this file's location
    # If running main.py directly from root, this might need adjustment or try/except
    # for different execution contexts.
    from utils.helpers import get_project_root, setup_logging
    _project_root_early = get_project_root() # Determine root path once
    # Initial minimal logging setup - will be reconfigured after loading config.yaml
    setup_logging({
        'logging': { # Structure expected by setup_logging
            'log_level_console': 'INFO',
            'log_level_file': 'WARNING', # Don't log DEBUG to file initially
            'log_dir': os.path.join(_project_root_early, 'logs'),
            'log_file_name': 'app_startup.log' # Use a distinct startup log? Or reuse main one?
        }
    })
    logging.info("Basic logging initialized before config load.")
except ImportError as e_util:
    logging.basicConfig(level=logging.INFO) # Absolute fallback logging
    logging.error(f"Failed to import utils.helpers: {e_util}. Logging/Root finding may be limited.")
    print(f"WARNING: Failed import from utils.helpers: {e_util}. Proceeding with basic logging.", file=sys.stderr)
    # Define fallback if get_project_root wasn't imported
    if _project_root_early is None:
         _project_root_early = os.path.dirname(os.path.abspath(__file__)) # Use main.py location as fallback root
         logging.warning(f"Falling back to project root: {_project_root_early}")
    # Define setup_logging as no-op if it failed import
    if 'setup_logging' not in globals():
        setup_logging = lambda *args, **kwargs: None
except Exception as e_log_setup:
    logging.basicConfig(level=logging.INFO) # Absolute fallback logging
    logging.error(f"Failed basic logging setup: {e_log_setup}", exc_info=True)
    print(f"WARNING: Failed basic logging setup: {e_log_setup}", file=sys.stderr)
    if 'setup_logging' not in globals():
        setup_logging = lambda *args, **kwargs: None


# --- Now Import Main Application Components ---
# Wrap these imports as they depend on the project structure being correct
try:
    from ui.main_window import MainWindow
    from ui.theme import ThemeManager # ThemeManager needed early if theme set before MainWindow init
    # Import other necessary top-level components if any
except ImportError as e_app_import:
    # Log critical error using the basic logging that should be set up
    logging.critical(f"Failed to import application components (MainWindow, ThemeManager, etc.): {e_app_import}", exc_info=True)
    print(f"[CRITICAL ERROR] Failed to import application components: {e_app_import}")
    print("Ensure the project structure is correct and all modules exist relative to the project root.")
    print(f"Determined Project Root: {_project_root_early}")
    # Attempt to show GUI message box (QApplication might not be fully running yet)
    try:
        if _QT_AVAILABLE:
            app_temp = QApplication.instance() or QApplication([]) # Get/create instance
            QMessageBox.critical(None, "Application Structure Error",
                                 f"Failed to import component: {e_app_import}.\nCheck project files and console output.")
    except Exception: pass # Ignore if message box fails
    sys.exit(1)
except Exception as e_imports:
    logging.critical(f"Unexpected error during application component imports: {e_imports}", exc_info=True)
    print(f"[CRITICAL ERROR] Unexpected error during application imports: {e_imports}")
    sys.exit(1)


# --- Configuration Loading ---
def load_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file, merging it over robust defaults.
    Uses deep copy to prevent modifying default structures.
    """
    # Define robust default values INSIDE the function to ensure they are fresh each time
    default_config = {
        'appearance': {
             'default_theme': 'dark_cosmic',
             'plotting': {
                  'matplotlib_style_dark': 'dark_background',
                  'matplotlib_style_light': 'seaborn-v0_8-notebook',
                  'nist_line_colormap': 'tab10'
              }
        },
        'processing': {
             'baseline': {'default_method': 'Polynomial', 'poly_order': 3, 'snip_iterations': 100, 'percentile': 10.0, 'max_iterations': 10, 'tolerance': 0.001},
             'smoothing': {'default_method': 'SavitzkyGolay', 'savitzky_golay': {'window_length': 11, 'polyorder': 3}},
             'denoising': {'default_method': 'Wavelet', 'wavelet': {'wavelet_type': 'db8', 'level': 4, 'mode': 'soft', 'threshold_sigma_factor': 3.0}},
             'noise_analysis': {}
        },
        'peak_detection': {
             'default_method': 'ScipyFindPeaks',
             'scipy_find_peaks': {'relative_height_percent': 5.0, 'min_distance_points': 5, 'width_points': 0, 'prominence': 0.0}
        },
        'peak_fitting': {
             'profiles_to_fit': ['Gaussian', 'Lorentzian', 'PseudoVoigt'],
             'roi_factor': 7.0, 'min_roi_width_nm': 0.1, 'min_roi_points': 5,
             'max_iterations': 2000, 'model_selection': 'AIC', 'baseline_mode': 'LocalLinear'
        },
        'atomic_data': {
            'partition_function_file': 'database/atomic_data/partition_functions.csv',
            'ionization_energy_file': 'database/atomic_data/ionization_energies.csv'
        },
        'nist_online_search': {
            'timeout_s': 20, 'query_delay_s': 1.5, 'default_search_tolerance_nm': 0.1
        },
        'cflibs': {
            'min_lines_for_boltzmann': 3
        },
        'machine_learning': {
            'preprocess_baseline_method': 'Polynomial', 'preprocess_scale_default': True,
            'pca': {'default_n_components': 3},
            'pls': {'default_n_components': 5, 'default_target_wl': 404.58},
            'RandomForest': {'n_estimators': 100},
            'GBT': {'n_estimators': 100, 'learning_rate': 0.1}
        },
        'file_io': {
            'default_delimiter': '\t', 'default_comment_char': '#', 'default_float_format': '%.6g'
        },
        'logging': {
            'log_level_console': 'INFO', 'log_level_file': 'DEBUG',
            'log_dir': 'logs', 'log_file_name': 'libs_cosmic_forge.log',
            'log_max_bytes': 5242880, 'log_backup_count': 4
        },
        'application': {
            'remember_window_state': True
        }
    }

    # Helper for deep merging dictionaries
    def merge_dicts(target: Dict, source: Dict):
        for key, value in source.items():
             if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                  # Recursively merge nested dictionaries
                  merge_dicts(target[key], value)
             elif isinstance(value, list) and key in target and isinstance(target[key], list):
                  # Simple list merge: source replaces target (or customize as needed)
                  target[key] = value
             else:
                  # Assign value (overwrites or adds)
                  target[key] = value

    try:
        # Start with a deep copy of defaults to avoid modifying the original structure
        final_config = copy.deepcopy(default_config)

        if not os.path.exists(config_path):
            logging.warning(f"Config file not found: {config_path}. Using default settings.")
            return final_config # Return the deep copy of defaults

        with open(config_path, 'r', encoding='utf-8') as f:
            config_loaded = yaml.safe_load(f)

        if not isinstance(config_loaded, dict):
             logging.warning(f"Config file '{config_path}' is empty or not a valid dictionary. Using default settings.")
             return final_config # Return defaults if file content is invalid

        # Perform the deep merge
        merge_dicts(final_config, config_loaded)

        logging.info(f"Configuration loaded successfully from {config_path} and merged with defaults.")
        return final_config

    except yaml.YAMLError as e_yaml:
        logging.error(f"Error parsing config file '{config_path}': {e_yaml}", exc_info=True)
        # Attempt to show GUI warning before returning defaults
        try:
            if _QT_AVAILABLE: QMessageBox.warning(None, "Config Error", f"Error parsing config.yaml:\n{e_yaml}\n\nUsing default settings.")
        except Exception: pass
        return copy.deepcopy(default_config) # Return defaults on YAML error
    except Exception as e_load:
        logging.error(f"Error loading or merging configuration from '{config_path}': {e_load}", exc_info=True)
        try:
            if _QT_AVAILABLE: QMessageBox.warning(None, "Config Error", f"Failed to load config.yaml:\n{e_load}\n\nUsing default settings.")
        except Exception: pass
        return copy.deepcopy(default_config) # Return defaults on other errors

# --- Global Exception Hook ---
_app_instance_for_hook = None # Store app instance for hook access

def global_exception_hook(exctype, value, tb):
    """Catches unhandled exceptions, logs them, attempts GUI message, and exits."""
    global _app_instance_for_hook
    # Format traceback first
    tb_text = "".join(traceback.format_exception(exctype, value, tb))
    error_msg_short = f"{exctype.__name__}: {value}"
    error_msg_long = f"A critical unhandled error occurred:\n\n{error_msg_short}"

    # Log the critical error immediately
    logging.critical("--- UNHANDLED EXCEPTION CAUGHT ---")
    logging.critical(error_msg_short, exc_info=(exctype, value, tb)) # Log with traceback info
    logging.critical("--- END UNHANDLED EXCEPTION ---")

    # Attempt to show GUI message box if possible
    gui_message_shown = False
    if _QT_AVAILABLE and _app_instance_for_hook:
        try:
            # Use a fresh QMessageBox instance
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Icon.Critical)
            error_box.setWindowTitle("Unhandled Application Error")
            error_box.setText(error_msg_long)
            error_box.setDetailedText(tb_text) # Full traceback in details section
            error_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            error_box.exec()
            gui_message_shown = True
        except Exception as hook_e:
            logging.error(f"Error within the global exception hook trying to show QMessageBox: {hook_e}", exc_info=True)

    # If GUI message failed or wasn't possible, ensure message goes to stderr
    if not gui_message_shown:
        print("\n--- UNHANDLED APPLICATION ERROR ---", file=sys.stderr)
        print(error_msg_long, file=sys.stderr)
        print("\nFull Traceback:\n", tb_text, file=sys.stderr)
        print("-----------------------------------", file=sys.stderr)

    # --- Attempt to save critical settings before forced exit ---
    emergency_save_done = False
    if _app_instance_for_hook and hasattr(_app_instance_for_hook, 'main_window'):
        main_win_ref = getattr(_app_instance_for_hook, 'main_window', None)
        if main_win_ref and hasattr(main_win_ref, '_save_persistent_settings'):
            logging.warning("Attempting emergency save of persistent settings before exiting...")
            try:
                main_win_ref._save_persistent_settings() # Try saving geometry, theme etc.
                logging.info("Emergency settings save successful.")
                emergency_save_done = True
            except Exception as e_hook_save:
                logging.error(f"Error during emergency save in exception hook: {e_hook_save}", exc_info=True)

    # --- Force Exit ---
    logging.critical("Exiting application forcefully due to unhandled error.")
    print("Application will now exit.", file=sys.stderr)
    os._exit(1) # Use os._exit for immediate exit without further cleanup, which might fail


# --- Main Application Execution ---
def run_app():
    """Initializes and runs the main application event loop."""
    global _app_instance_for_hook
    app = None # Define app outside try block for potential use in except/finally
    main_window = None
    root_dir = _project_root_early # Use the root determined before imports
    config_path = os.path.join(root_dir, 'config.yaml')

    # Assign the exception hook EARLY, after basic logging is set up
    sys.excepthook = global_exception_hook
    logging.info("Global exception hook assigned.")

    try:
        # 1. Load Configuration FIRST
        config = load_config(config_path)
        # load_config logs its own success/warnings/errors

        # 2. Setup Logging AGAIN (fully configured using loaded config)
        # Pass the 'logging' section from the merged config
        setup_logging(config.get('logging', {}))
        logging.info("Logging reconfigured using settings from config.yaml.")
        logging.info(f"--- Starting {APP_NAME} v{APP_VERSION} ---")
        logging.info(f"Project Root: {root_dir}")
        logging.debug(f"Loaded Config Keys: {list(config.keys())}") # Log keys, not values

        # 3. Initialize QApplication (Ensure only one instance exists)
        logging.debug("Initializing QApplication...")
        app = QApplication.instance() # Check if already exists (e.g., from tests)
        if app is None:
            logging.debug("No QApplication instance found, creating a new one.")
            app = QApplication(sys.argv)
        else:
            logging.debug("Reusing existing QApplication instance.")
        _app_instance_for_hook = app # Make instance available to hook


        # 4. Create and Show Main Window
        logging.debug("Creating MainWindow...")
        try:
            main_window = MainWindow(config) # Pass full config dictionary
            # Store reference on app instance for hook access (optional but convenient)
            app.main_window = main_window
            logging.debug("MainWindow instance created.")
            main_window.show()
            logging.info("MainWindow shown.")
        except Exception as e_win:
            # Log the specific window creation error critically
            logging.critical(f"Failed to initialize or show the main window: {e_win}", exc_info=True)
            # Show a message box (QApplication should exist now)
            QMessageBox.critical(None, "Application Startup Error",
                                 f"Failed to create the main application window:\n{e_win}\n\n"
                                 "Application cannot continue. Check logs for details.")
            sys.exit(1) # Exit if the main window fails critically

        # 5. Start Event Loop
        logging.info("Application startup successful. Entering Qt event loop.")
        exit_code = app.exec()
        logging.info(f"Application event loop finished. Exiting with code {exit_code}.")
        sys.exit(exit_code) # Use the exit code from the event loop

    except SystemExit as e_exit:
        # Catch sys.exit() explicitly to log appropriately
        if e_exit.code == 0:
             logging.info(f"Application exiting normally (Code: {e_exit.code}).")
        else:
             logging.warning(f"Application exiting with non-zero code: {e_exit.code}.")
        # Do not raise here, let the exit proceed
        sys.exit(e_exit.code) # Ensure exit happens

    except Exception as e_main:
        # This catch block is a final fallback *if* the global exception hook somehow fails
        # or if the error happens very early before the hook is assigned.
        logging.critical(f"Unexpected error in main run_app scope: {e_main}", exc_info=True)
        print(f"FATAL ERROR (main.py guard): {e_main}\n{''.join(traceback.format_exc())}", file=sys.stderr)
        try:
            # Attempt message box one last time if app might exist
            if _QT_AVAILABLE and QApplication.instance():
                QMessageBox.critical(None, "Fatal Error", f"Fatal error during application run:\n{e_main}\nApp will exit.")
        except Exception: pass # Ignore errors showing final message box
        finally:
            sys.exit(1) # Force exit


# --- Entry Point Guard ---
if __name__ == "__main__":
    # The global exception hook is assigned above
    run_app() # Call the main execution function