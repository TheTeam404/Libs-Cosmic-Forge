# -*- coding: utf-8 -*-
import logging
import os
import sys
import traceback
from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np  # Added import
import pandas as pd
from PyQt6.QtCore import (
    QSize, Qt, pyqtSlot, QSettings, QByteArray, QPoint, QCoreApplication,
    QUrl, QProcess, QStandardPaths  # Ensure QStandardPaths is imported
)
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QCursor, QDesktopServices
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QStatusBar, QMenuBar,
    QMessageBox, QApplication, QFileDialog, QDockWidget, QToolBar ,QMenu
)

# Local Imports
# Assuming these paths are correct relative to main_window.py execution context
# If running from project root, paths might need adjustment if not using sys.path hacks
try:
    from utils.helpers import get_project_root
    from ui.theme import ThemeManager
    from core.data_models import Spectrum, Peak, NISTMatch, FitResult
    from core.file_io import (
        load_spectrum_from_file, save_spectrum_data, save_peak_list,
        save_nist_matches, save_dataframe
    )
    from core.session_manager import SessionManager
    from ui.views.plot_widget import SpectrumPlotWidget
    from ui.views.control_panel_view import ProcessingControlPanel
    from ui.views.peak_controls_view import PeakDetectionControlPanel
    from ui.views.peak_fitting_controls_view import PeakFittingControlPanel
    from ui.views.peak_list_view import PeakListView
    from ui.views.nist_search_view import NistSearchView
    from ui.views.boltzmann_plot_view import BoltzmannPlotView
    from ui.views.ml_analysis_view import MLAnalysisView
    from ui.views.cf_libs_view import CfLibsView
    from ui.external_script_runner import ExternalScriptRunnerDialog
    from core.processing import baseline_poly, baseline_snip, smooth_savitzky_golay
    from core.peak_detector import detect_peaks_scipy
    from core.peak_fitter import fit_peak
    from core.nist_manager import search_online_nist # Ensure this exists and is correct
    from core.cflibs import calculate_electron_density_saha, calculate_cf_libs_conc # Ensure this exists
except ImportError as e:
    logging.critical(f"Failed to import necessary modules: {e}. Check PYTHONPATH or relative paths.", exc_info=True)
    # Attempting to find project root if get_project_root failed initially
    try:
        # Simple fallback - might not be robust
        project_root_fallback = Path(__file__).parent.parent.resolve()
        sys.path.insert(0, str(project_root_fallback))
        logging.warning(f"Attempting import retry with project root: {project_root_fallback}")
        # Re-try imports specifically needed here if possible, or inform user
        from utils.helpers import get_project_root # Retry this one for icon loading
        logging.info("Successfully imported utils.helpers after path adjustment.")
    except ImportError:
        logging.critical("Failed to adjust path and import helpers. Icon loading will fail.")
        # Define a dummy get_project_root if necessary for execution continuation
        def get_project_root(): return "." # Non-functional placeholder
    # Decide whether to raise the original error or exit
    # raise e # Re-raise the original error to halt execution
    sys.exit(f"Import Error: {e}") # Or exit cleanly


# --- Constants ---
APP_VERSION = "0.2.3" # Incremented version for refactor
ORGANIZATION_NAME = "CosmicForgeDev"
APPLICATION_NAME = "LIBSForge"
DEFAULT_THEME = "dark_cosmic"

# --- Helper Functions ---
def get_icon(name: str) -> QIcon:
    """Loads an icon from the assets folder or falls back to a theme icon."""
    try:
        project_root = get_project_root() # Call the helper function
        icon_path = os.path.join(project_root, "assets", "icons", name)
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        else:
            # Fallback to theme icon (e.g., "load_spectrum.png" -> "document-open")
            theme_name = name.split('.')[0].replace('_', '-')
            fallback_icon = QIcon.fromTheme(theme_name, QIcon()) # Provide default QIcon
            if fallback_icon.isNull():
                logging.warning(f"Icon '{name}' not found in assets path '{icon_path}' or theme '{theme_name}'.")
            return fallback_icon
    except Exception as e:
         logging.error(f"Error getting icon '{name}': {e}. Returning empty icon.", exc_info=True)
         return QIcon()


# --- Main Window Class ---
class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config: dict):
        """Initializes the main window, UI components, and internal state."""
        super().__init__()
        self.config = config

        # --- Application Info & Settings ---
        QCoreApplication.setOrganizationName(ORGANIZATION_NAME)
        QCoreApplication.setApplicationName(APPLICATION_NAME)
        self.settings = QSettings() # Use instance variable for settings

        # --- Core Components ---
        self.theme_manager = ThemeManager(QApplication.instance(), self.config)
        self.session_manager = SessionManager(self) # Pass self (MainWindow instance)

        # --- Application State ---
        self.current_spectrum: Optional[Spectrum] = None
        self.multi_spectra: List[Spectrum] = []
        self.detected_peaks: List[Peak] = []
        self.nist_matches: List[NISTMatch] = []
        self.plasma_temp_k: Optional[float] = None
        self.electron_density_cm3: Optional[float] = None
        self.partition_functions: Optional[Dict[str, float]] = None # Placeholder
        self.ionization_energies: Optional[Dict[str, float]] = None # Placeholder
        self.boltzmann_plot_data: Optional[pd.DataFrame] = None
        self.cf_libs_concentrations: Optional[pd.DataFrame] = None

        # --- Configuration & Defaults ---
        self.default_delimiter = config.get('file_io', {}).get('default_delimiter', '\t')
        self.default_comment = config.get('file_io', {}).get('default_comment_char', '#')
        self.remember_window_state = config.get('application', {}).get('remember_window_state', True)

        # --- UI State ---
        self._is_busy: bool = False

        # --- Debugging Print (Remove once fixed) ---
        print("DEBUG: Attributes available on self before _get_default_directory call:")
        print(dir(self))
        # --- END Debugging Print ---

        # --- Call _get_default_directory (Ensure it's defined BEFORE this line) ---
        try:
            default_dir = self._get_default_directory() # Call the method
        except AttributeError as e:
             # This should ideally not happen if the method is defined and indented correctly
             logging.critical(f"FATAL: Still cannot find _get_default_directory! Error: {e}", exc_info=True)
             # Fallback to a very basic default if the method fails catastrophically
             default_dir = os.path.expanduser("~")
             QMessageBox.critical(self, "Initialization Error",
                                  "Could not determine default directory. Falling back to home.\n"
                                  "Please check logs. Error: " + str(e))
             # Consider exiting if this is critical: sys.exit(1)
        except Exception as e:
             logging.error(f"Error calling _get_default_directory: {e}", exc_info=True)
             default_dir = os.path.expanduser("~") # Fallback
             QMessageBox.warning(self, "Initialization Warning",
                                 f"Could not determine default directory. Falling back to home.\nError: {e}")

        # --- Assign last load/save dirs using the result ---
        self._last_save_dir: str = default_dir
        self._last_load_dir: str = default_dir

        self.external_process: Optional[QProcess] = None

        # --- UI Elements (initialized in _init_ui) ---
        self.plot_widget: Optional[SpectrumPlotWidget] = None
        self.processing_panel: Optional[ProcessingControlPanel] = None
        self.peak_detection_panel: Optional[PeakDetectionControlPanel] = None
        self.peak_fitting_panel: Optional[PeakFittingControlPanel] = None
        self.peak_list_view: Optional[PeakListView] = None
        self.nist_search_view: Optional[NistSearchView] = None
        self.boltzmann_view: Optional[BoltzmannPlotView] = None
        self.cf_libs_view: Optional[CfLibsView] = None
        self.ml_view: Optional[MLAnalysisView] = None
        self.docks: Dict[str, QDockWidget] = {}
        self.status_label: Optional[QLabel] = None
        self.panels_menu: Optional[QMenu] = None
        self.theme_actions: Dict[str, QAction] = {}
        # Action attributes (defined in menu/toolbar creation)
        self.load_action: Optional[QAction] = None
        self.load_multi_action: Optional[QAction] = None
        self.load_session_action: Optional[QAction] = None # This holds the QAction object
        self.save_session_action: Optional[QAction] = None
        self.save_processed_action: Optional[QAction] = None
        self.save_peaks_action: Optional[QAction] = None
        self.save_nist_action: Optional[QAction] = None
        self.save_boltzmann_action: Optional[QAction] = None
        self.save_conc_action: Optional[QAction] = None
        self.save_plot_action: Optional[QAction] = None
        self.save_plot_toolbar_action: Optional[QAction] = None
        self.reset_zoom_action: Optional[QAction] = None

        # --- Initialization Steps ---
        self._init_ui()
        self._connect_signals()
        self._load_persistent_settings() # Handles theme and window state
        self.update_status(f"Welcome to LIBS Forge v{APP_VERSION}!")
        logging.info(f"LIBS Forge v{APP_VERSION} initialized.")
        # Inside the MainWindow class definition:


    # ***** MOVED METHOD DEFINITION *****
    # Define helper methods like this BEFORE they are called in __init__
    # Ensure this is indented correctly to be part of the MainWindow class
    def _get_default_directory(self) -> str:
        """Returns the user's default documents or home directory."""
        # Ensure QStandardPaths and os are imported at the top of the file
        try:
            docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            logging.debug(f"QStandardPaths DocumentsLocation returned: '{docs}'")
            # Return Documents location if found and not empty, otherwise fallback to home
            return docs if docs else os.path.expanduser("~")
        except Exception as e:
             # Catch potential errors during QStandardPaths call
             logging.error(f"Error getting QStandardPaths.DocumentsLocation: {e}. Falling back to home dir.", exc_info=True)
             return os.path.expanduser("~")


    # ***** Ensure this method is indented correctly *****
    # It seems to be indented correctly in your original code, but double-check
    def _update_save_actions_state(self):
        """
        Enables or disables save-related menu actions based on whether
        data is currently loaded or available to be saved.
        """
        # --- Step 1: Determine if data is available for saving ---
        # More robust checks based on actual state variables
        has_spec = self.current_spectrum is not None
        has_proc = has_spec and self.current_spectrum.processed_intensity is not None
        has_peaks = bool(self.detected_peaks)
        has_nist = bool(self.nist_matches)
        has_boltz_data = self.boltzmann_plot_data is not None and not self.boltzmann_plot_data.empty
        has_conc_data = self.cf_libs_concentrations is not None and not self.cf_libs_concentrations.empty
        has_multi = bool(self.multi_spectra)
        can_save_session = has_spec or has_multi # Can save session if either single or multi spectra are loaded

        # --- Step 2: Enable/Disable the QAction objects ---
        # Use the specific action names defined in _create_file_menu
        if hasattr(self, 'save_processed_action') and self.save_processed_action:
            self.save_processed_action.setEnabled(has_proc)
        if hasattr(self, 'save_peaks_action') and self.save_peaks_action:
            self.save_peaks_action.setEnabled(has_peaks)
        if hasattr(self, 'save_nist_action') and self.save_nist_action:
            self.save_nist_action.setEnabled(has_nist)
        if hasattr(self, 'save_boltzmann_action') and self.save_boltzmann_action:
             self.save_boltzmann_action.setEnabled(has_boltz_data)
        if hasattr(self, 'save_conc_action') and self.save_conc_action:
             self.save_conc_action.setEnabled(has_conc_data)
        if hasattr(self, 'save_plot_action') and self.save_plot_action:
             self.save_plot_action.setEnabled(has_spec) # Enable saving plot if spectrum exists
        if hasattr(self, 'save_plot_toolbar_action') and self.save_plot_toolbar_action:
             self.save_plot_toolbar_action.setEnabled(has_spec)
        if hasattr(self, 'save_session_action') and self.save_session_action:
             self.save_session_action.setEnabled(can_save_session)

        logging.debug(f"Updated save actions state: Proc={has_proc}, Peaks={has_peaks}, NIST={has_nist}, Boltz={has_boltz_data}, Conc={has_conc_data}, Plot={has_spec}, Session={can_save_session}")


    # --- Initialization & Setup Methods ---

    def _init_ui(self):
        """Sets up the main UI elements of the window."""
        self.setWindowTitle(f"LIBS Forge v{APP_VERSION}")
        self._setup_geometry()
        self._setup_icon()
        self._setup_status_bar()
        self._setup_central_widget()
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_dock_widgets()
        self.setDockNestingEnabled(True)

    # --- MOVED _get_default_directory method definition EARLIER (before __init__ call) ---

    def _setup_geometry(self):
        """Sets the initial size and position of the window."""
        try:
            screen = QApplication.primaryScreen()
            if not screen:
                 logging.warning("Could not get primary screen. Using default size.")
                 self.resize(1400, 900)
                 return

            available_geo = screen.availableGeometry()
            width = int(available_geo.width() * 0.85)
            height = int(available_geo.height() * 0.85)
            x_pos = int(available_geo.left() + (available_geo.width() - width) / 2) # Center horizontally
            y_pos = int(available_geo.top() + (available_geo.height() - height) / 2) # Center vertically
            self.setGeometry(x_pos, y_pos, width, height)
            logging.debug(f"Initial geometry set based on screen: {x_pos},{y_pos} {width}x{height}")
        except Exception as e:
            logging.warning(f"Could not determine screen geometry, using default size. Error: {e}", exc_info=True)
            self.resize(1400, 900) # Default fallback size

    def _setup_icon(self):
        """Sets the application window icon."""
        self.setWindowIcon(get_icon("app_icon.png")) # Use helper

    def _setup_status_bar(self):
        """Creates and configures the status bar."""
        statusBar = QStatusBar(self) # Pass parent
        self.setStatusBar(statusBar)
        self.status_label = QLabel("Initializing...")
        self.status_label.setObjectName("StatusLabel") # For styling
        statusBar.addPermanentWidget(self.status_label)

    def _setup_central_widget(self):
        """Sets the central widget (the main spectrum plot)."""
        self.plot_widget = SpectrumPlotWidget(self, self.config) # Pass self and config
        self.setCentralWidget(self.plot_widget)

    def _create_menu_bar(self):
        """Creates the main menu bar and its menus."""
        menubar = self.menuBar()
        self._create_file_menu(menubar)
        self._create_view_menu(menubar)
        self._create_tools_menu(menubar)
        self._create_help_menu(menubar)

    def _create_file_menu(self, menubar: QMenuBar):
        """Creates the 'File' menu."""
        file_menu = menubar.addMenu("&File")

        # Load Spectrum Action
        self.load_action = QAction(get_icon("load_spectrum.png"), "&Load Spectrum...", self)
        self.load_action.setShortcut(QKeySequence.StandardKey.Open)
        self.load_action.setStatusTip("Load a single spectrum file")
        self.load_action.triggered.connect(self.load_spectrum_action)
        file_menu.addAction(self.load_action)

        # Load Multiple Spectra Action
        self.load_multi_action = QAction(get_icon("load_multi_spectra.png"), "Load &Multiple Spectra...", self)
        self.load_multi_action.setStatusTip("Load multiple spectra for ML analysis or comparison")
        self.load_multi_action.triggered.connect(self._load_multiple_spectra_action)
        file_menu.addAction(self.load_multi_action)

        # Load Session Action
        self.load_session_action = QAction(get_icon("document-open.png"), "Load Session...", self) # Used custom icon name
        self.load_session_action.setShortcut("Ctrl+L")
        self.load_session_action.setStatusTip("Load a previously saved analysis session state")
        self.load_session_action.triggered.connect(self._on_load_session_triggered) # Connect to the correct slot
        file_menu.addAction(self.load_session_action)

        file_menu.addSeparator()

        # --- Save Submenu ---
        save_menu = file_menu.addMenu(get_icon("save_figure.png"), "&Save") # Use a generic save icon for submenu

        self.save_session_action = QAction(get_icon("document-save.png"), "Save Session...", self) # Used custom icon name
        self.save_session_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_session_action.setStatusTip("Save the current analysis state, data, and settings")
        self.save_session_action.triggered.connect(self._on_save_session_triggered) # Connects to correct slot
        save_menu.addAction(self.save_session_action)

        save_menu.addSeparator()

        self.save_processed_action = QAction("Processed Spectrum (.csv)", self)
        self.save_processed_action.setStatusTip("Save wavelength, raw intensity, and processed intensity data")
        self.save_processed_action.triggered.connect(lambda: self._save_action('processed_spectrum'))
        save_menu.addAction(self.save_processed_action)

        self.save_peaks_action = QAction("Peak List (.csv)", self)
        self.save_peaks_action.setStatusTip("Save detected and fitted peak parameters")
        self.save_peaks_action.triggered.connect(lambda: self._save_action('peaks'))
        save_menu.addAction(self.save_peaks_action)

        self.save_nist_action = QAction("NIST Matches (.csv)", self)
        self.save_nist_action.setStatusTip("Save the table of potential NIST line matches found")
        self.save_nist_action.triggered.connect(lambda: self._save_action('nist_matches'))
        save_menu.addAction(self.save_nist_action)

        self.save_boltzmann_action = QAction("Boltzmann Data (.csv)", self)
        self.save_boltzmann_action.setStatusTip("Save the data points used for the Boltzmann plot calculation")
        self.save_boltzmann_action.triggered.connect(lambda: self._save_action('boltzmann'))
        save_menu.addAction(self.save_boltzmann_action)

        self.save_conc_action = QAction("Concentrations (.csv)", self)
        self.save_conc_action.setStatusTip("Save calculated CF-LIBS concentrations")
        self.save_conc_action.triggered.connect(lambda: self._save_action('concentrations'))
        save_menu.addAction(self.save_conc_action)

        self.save_plot_action = QAction("Plot Image (.png, .svg)...", self)
        self.save_plot_action.setStatusTip("Save the current main plot view as an image file")
        self.save_plot_action.triggered.connect(lambda: self._save_action('plot'))
        save_menu.addAction(self.save_plot_action)

        self._update_save_actions_state() # Initialize save action states based on current (empty) state

        file_menu.addSeparator()

        exit_action = QAction(get_icon("exit.png"), "E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close) # Use built-in close slot
        file_menu.addAction(exit_action)

    def _create_view_menu(self, menubar: QMenuBar):
        """Creates the 'View' menu."""
        view_menu = menubar.addMenu("&View")

        # --- Themes Submenu ---
        theme_menu = view_menu.addMenu("Themes")
        self.theme_actions = {}
        available_themes = self.theme_manager.get_available_themes()
        if not available_themes:
             no_themes_action = QAction("No themes found", self)
             no_themes_action.setEnabled(False)
             theme_menu.addAction(no_themes_action)
        else:
             for theme_name in available_themes:
                 action = QAction(theme_name.replace('_', ' ').title(), self, checkable=True)
                 # Use lambda with default argument to capture current theme_name
                 action.triggered.connect(lambda checked, name=theme_name: self.change_theme(name))
                 self.theme_actions[theme_name] = action
                 theme_menu.addAction(action)
             self.update_theme_menu() # Set initial check state

        # --- Panels Submenu ---
        # Populated when docks are created in _create_dock_widgets
        self.panels_menu = view_menu.addMenu("Panels")

        view_menu.addSeparator()

        self.reset_zoom_action = QAction(get_icon("reset_zoom.png"), "Reset Zoom", self)
        self.reset_zoom_action.setShortcut("Ctrl+H") # Example shortcut
        self.reset_zoom_action.setStatusTip("Reset plot zoom and pan to the full view")
        # Connect using lambda for safety (plot_widget might not exist initially)
        self.reset_zoom_action.triggered.connect(
            lambda: self.plot_widget.toolbar.home() if (
                self.plot_widget and hasattr(self.plot_widget, 'toolbar') and self.plot_widget.toolbar
            ) else logging.warning("Reset Zoom triggered but plot widget/toolbar not available.")
        )
        view_menu.addAction(self.reset_zoom_action)

    def _create_tools_menu(self, menubar: QMenuBar):
        """Creates the 'Tools' menu."""
        tools_menu = menubar.addMenu("&Tools")

        fetch_nist_action = QAction(get_icon("download.png"), "Fetch NIST Data (Script)...", self)
        fetch_nist_action.setStatusTip("Run the nist_data_fetcher.py script to download atomic data")
        fetch_nist_action.triggered.connect(self.run_nist_fetcher)
        tools_menu.addAction(fetch_nist_action)

        build_data_action = QAction(get_icon("database.png"), "Build Atomic Data Files (Script)...", self)
        build_data_action.setStatusTip("Run the atomic_data_builder.py script (placeholder)")
        build_data_action.triggered.connect(self.run_atomic_data_builder)
        # build_data_action.setEnabled(False) # If it's truly a placeholder
        tools_menu.addAction(build_data_action)

    def _create_help_menu(self, menubar: QMenuBar):
        """Creates the 'Help' menu."""
        help_menu = menubar.addMenu("&Help")

        about_act = QAction(f"About LIBS Forge", self) # Version added in dialog
        about_act.setStatusTip(f"Show information about LIBS Forge v{APP_VERSION}")
        about_act.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_act)

        online_docs_act = QAction(get_icon("help-contents.png"), "Online Documentation", self) # Used custom icon name
        online_docs_act.setStatusTip("Open the online documentation (placeholder)")
        online_docs_act.triggered.connect(self._open_online_docs)
        # online_docs_act.setEnabled(False) # If no docs exist yet
        help_menu.addAction(online_docs_act)

    def _create_tool_bar(self):
        """Creates the main application toolbar."""
        toolbar = QToolBar("Main Toolbar", self) # Pass parent
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        # Add actions (ensure they exist from menu creation)
        if self.load_action:
            self.load_action.setToolTip("Load single spectrum (Ctrl+O)")
            toolbar.addAction(self.load_action)

        # Save Plot action specifically for toolbar
        self.save_plot_toolbar_action = QAction(get_icon("save_figure.png"), "Save Plot", self)
        self.save_plot_toolbar_action.setToolTip("Save the current plot as an image")
        self.save_plot_toolbar_action.triggered.connect(lambda: self._save_action('plot'))
        self.save_plot_toolbar_action.setEnabled(False) # Initially disabled until spectrum loaded
        toolbar.addAction(self.save_plot_toolbar_action)

        toolbar.addSeparator()

        # Add plot navigation actions from the plot widget's toolbar safely
        if self.plot_widget and hasattr(self.plot_widget, 'toolbar') and self.plot_widget.toolbar:
            # Find actions robustly by text/tooltip if possible
            nav_toolbar_actions = self.plot_widget.toolbar.actions()
            home_action = next((a for a in nav_toolbar_actions if "Home" in a.toolTip()), None) # Reset is often called 'Home'
            pan_action = next((a for a in nav_toolbar_actions if "Pan" in a.toolTip()), None)
            zoom_action = next((a for a in nav_toolbar_actions if "Zoom" in a.toolTip()), None)

            # Use the specific Reset Zoom action created for the View menu
            if self.reset_zoom_action:
                 self.reset_zoom_action.setToolTip("Reset Zoom (Ctrl+H)")
                 toolbar.addAction(self.reset_zoom_action)
            elif home_action: # Fallback to plot toolbar's home if reset_zoom_action is missing
                 home_action.setIcon(get_icon("reset_zoom.png")) # Standardize icon
                 home_action.setToolTip("Reset Zoom (Ctrl+H)")
                 toolbar.addAction(home_action)
                 logging.warning("Using plot toolbar's Home action for Reset Zoom.")
            else:
                 logging.warning("Could not find Reset Zoom/Home action in plot toolbar.")

            if pan_action:
                 pan_action.setIcon(get_icon("pan.png"))
                 pan_action.setToolTip("Pan/Move Plot (Left-Click & Drag or MMB)")
                 toolbar.addAction(pan_action)
            else:
                 logging.warning("Could not find Pan action in plot toolbar.")

            if zoom_action:
                 zoom_action.setIcon(get_icon("zoom.png"))
                 zoom_action.setToolTip("Zoom Box (Right-Click & Drag)")
                 toolbar.addAction(zoom_action)
            else:
                 logging.warning("Could not find Zoom action in plot toolbar.")
        else:
             logging.warning("Plot widget or its toolbar not available when creating main toolbar.")


    def _create_dock_widgets(self):
        """Creates and arranges all the dockable panels."""
        self.docks = {}
        left_area = Qt.DockWidgetArea.LeftDockWidgetArea
        right_area = Qt.DockWidgetArea.RightDockWidgetArea
        bottom_area = Qt.DockWidgetArea.BottomDockWidgetArea

        # Define features once
        dock_features = (QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                         QDockWidget.DockWidgetFeature.DockWidgetClosable)

        def add_dock(name: str, title: str, widget: QWidget, area: Qt.DockWidgetArea, shortcut_num: int, tabify_with: Optional[QDockWidget] = None, initial_enabled: bool = False) -> QDockWidget:
            """Helper function to create and add a dock widget."""
            if not isinstance(widget, QWidget):
                 logging.error(f"Cannot add dock '{name}'. Provided widget is not a QWidget (type: {type(widget)}).")
                 return None # Return None if widget is invalid

            dock = QDockWidget(title, self)
            dock.setObjectName(f"{name}Dock")
            dock.setWidget(widget)
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas) # Allow all areas
            dock.setFeatures(dock_features)
            self.addDockWidget(area, dock)
            self.docks[name] = dock

            # Add toggle action to View -> Panels menu
            toggle_action = dock.toggleViewAction()
            toggle_action.setText(title) # Use the dock title for the menu item
            toggle_action.setShortcut(f"Ctrl+{shortcut_num}")
            if self.panels_menu:
                self.panels_menu.addAction(toggle_action)
            else:
                # This might happen if menu creation order changes
                logging.error("Panels menu ('View -> Panels') not initialized before adding dock toggle action.")

            # Set initial enabled state of the *contained widget*
            widget.setEnabled(initial_enabled)

            # Tabify with the previously created dock if specified
            if tabify_with:
                self.tabifyDockWidget(tabify_with, dock)

            return dock

        # Create panels (widgets first, ensure they are QWidget subclasses)
        try:
            self.processing_panel = ProcessingControlPanel(self.config)
            self.peak_detection_panel = PeakDetectionControlPanel(self.config)
            self.peak_fitting_panel = PeakFittingControlPanel(self.config)
            self.peak_list_view = PeakListView(self)
            self.nist_search_view = NistSearchView(self.config, self)
            self.boltzmann_view = BoltzmannPlotView(self.config, self)
            self.cf_libs_view = CfLibsView(self.config, self)
            self.ml_view = MLAnalysisView(self.config, self)
        except Exception as e:
             logging.critical(f"Failed to instantiate one or more panel widgets: {e}", exc_info=True)
             QMessageBox.critical(self, "UI Initialization Error", f"Failed to create control panels:\n{e}")
             # Cannot proceed if panels fail, maybe exit?
             # sys.exit(1)
             return # Stop dock creation

        # Add docks (specifying area, shortcut number, and optional tabbing)
        # Left docks (tabbed)
        proc_dock = add_dock('processing', 'Processing', self.processing_panel, left_area, 1, initial_enabled=False)
        detect_dock = add_dock('detection', 'Detection', self.peak_detection_panel, left_area, 2, proc_dock, initial_enabled=False)
        fit_dock = add_dock('fitting', 'Fitting', self.peak_fitting_panel, left_area, 3, detect_dock, initial_enabled=False)

        # Right docks (tabbed)
        list_dock = add_dock('peak_list', 'Peak List', self.peak_list_view, right_area, 4, initial_enabled=False)
        nist_dock = add_dock('nist_search', 'NIST Search', self.nist_search_view, right_area, 5, list_dock, initial_enabled=False)

        # Bottom docks (tabbed)
        boltzmann_dock = add_dock('boltzmann', 'Boltzmann', self.boltzmann_view, bottom_area, 6, initial_enabled=False)
        cflibs_dock = add_dock('cflibs', 'CF-LIBS', self.cf_libs_view, bottom_area, 7, boltzmann_dock, initial_enabled=False)
        ml_dock = add_dock('ml_analysis', 'ML Analysis', self.ml_view, bottom_area, 8, cflibs_dock, initial_enabled=False)

        # Set initial visibility/focus for the first tab in each group
        # Check if docks were created successfully before raising
        if proc_dock: proc_dock.raise_()
        if list_dock: list_dock.raise_()
        if boltzmann_dock: boltzmann_dock.raise_()


    def _connect_signals(self):
        """Connects signals from UI components to slots in the main window."""
        # Check if panels exist before connecting signals (robustness)
        if self.processing_panel and hasattr(self.processing_panel, 'process_triggered'):
            self.processing_panel.process_triggered.connect(self.handle_process_request)
        else: logging.warning("Processing panel or its signal not found for connection.")

        if self.peak_detection_panel and hasattr(self.peak_detection_panel, 'detect_peaks_triggered'):
            self.peak_detection_panel.detect_peaks_triggered.connect(self.handle_peak_detection_request)
        else: logging.warning("Peak detection panel or its signal not found.")

        if self.peak_fitting_panel:
             if hasattr(self.peak_fitting_panel, 'fit_peaks_triggered'):
                 self.peak_fitting_panel.fit_peaks_triggered.connect(self.handle_peak_fitting_request)
             if hasattr(self.peak_fitting_panel, 'refit_single_peak_requested'):
                 self.peak_fitting_panel.refit_single_peak_requested.connect(self.handle_refit_single_peak)
             if hasattr(self.peak_fitting_panel, 'show_specific_fit'):
                 self.peak_fitting_panel.show_specific_fit.connect(self.handle_show_specific_fit)
        else: logging.warning("Peak fitting panel not found for connections.")

        if self.peak_list_view and hasattr(self.peak_list_view, 'peak_selected'):
            self.peak_list_view.peak_selected.connect(self.handle_peak_selection)
        else: logging.warning("Peak list view or its signal not found.")

        if self.plot_widget and hasattr(self.plot_widget, 'peak_clicked'):
            self.plot_widget.peak_clicked.connect(self.handle_peak_plot_click)
        else: logging.warning("Plot widget or its signal not found.")

        if self.nist_search_view:
             # Check if the specific signal exists
             if hasattr(self.nist_search_view, 'online_results_obtained'):
                 self.nist_search_view.online_results_obtained.connect(self._handle_nist_search_results)
             else:
                 logging.warning("NistSearchView exists but does not have the 'online_results_obtained' signal.")
        else: logging.warning("NIST search view not found for connections.")

        if self.boltzmann_view:
             if hasattr(self.boltzmann_view, 'populate_lines_requested'):
                 self.boltzmann_view.populate_lines_requested.connect(self.handle_boltzmann_populate_request)
             if hasattr(self.boltzmann_view, 'calculation_complete'):
                 self.boltzmann_view.calculation_complete.connect(self._handle_boltzmann_result)
        else: logging.warning("Boltzmann view not found for connections.")

        if self.cf_libs_view:
            if hasattr(self.cf_libs_view, 'calculate_ne_requested'):
                 self.cf_libs_view.calculate_ne_requested.connect(self.handle_ne_calculation_request)
            if hasattr(self.cf_libs_view, 'calculate_conc_requested'):
                 self.cf_libs_view.calculate_conc_requested.connect(self.handle_conc_calculation_request)
        else: logging.warning("CF-LIBS view not found for connections.")

        if self.ml_view and hasattr(self.ml_view, 'status_update'):
             self.ml_view.status_update.connect(self.update_status) # Example connection
        else: logging.warning("ML Analysis view or its signal not found.")

    # --- UI Update & Helper Methods ---

    def set_busy(self, busy: bool, message: str = "Working..."):
        """Sets the application to a busy state (e.g., wait cursor, status message)."""
        self._is_busy = busy
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.update_status(message, timeout=0) # Persistent message while busy
        else:
            QApplication.restoreOverrideCursor()
            self.update_status("Ready.") # Default message when not busy
        QApplication.processEvents() # Ensure UI updates (cursor, status) immediately

    def update_status(self, message: str, timeout: int = 0):
        """Updates the status bar message."""
        if self.status_label:
            logging.debug(f"Status Update: {message}")
            if timeout == 0:
                 # Set persistent message on the custom label
                 self.status_label.setText(message)
                 # Clear any temporary message shown by statusBar().showMessage()
                 if self.statusBar(): self.statusBar().clearMessage()
            else:
                 # Use temporary message display on the status bar itself
                 if self.statusBar():
                      self.statusBar().showMessage(message, timeout * 1000) # timeout is in ms
                      # Keep the persistent label showing something reasonable? Or clear it?
                      # self.status_label.setText("Ready.") # Option: Reset persistent label
                 else: # Fallback if status bar itself is missing
                      self.status_label.setText(message)

        else:
             # This should not happen after initialization
             logging.warning(f"Status label not initialized. Message ignored: {message}")


    def show_about_dialog(self):
        """Displays the 'About' dialog box."""
        about_text = f"""
        <h2>LIBS Forge v{APP_VERSION}</h2>
        <p>Advanced LIBS Analysis Suite.</p>
        <p>Developed by: {ORGANIZATION_NAME}</p>
        <p>Built with Python {sys.version_info.major}.{sys.version_info.minor} & PyQt{PYQT_VERSION_STR if 'PYQT_VERSION_STR' in locals() else '6'}</p>
        <p><b>Core Libraries:</b></p>
        <ul>
            <li>Matplotlib</li>
            <li>SciPy</li>
            <li>Pandas</li>
            <li>NumPy</li>
            <li>Astroquery (for NIST)</li>
            <li>Scikit-learn</li>
            <li>PyYAML</li>
        </ul>
        <p><i>Further licensing details TBD.</i></p>
        <hr>
        <p>Project Root: {get_project_root()}</p>
        """
        # Get PyQt version dynamically if possible
        try:
            from PyQt6.QtCore import PYQT_VERSION_STR
        except ImportError:
            PYQT_VERSION_STR = "N/A" # Fallback

        QMessageBox.about(self, f"About LIBS Forge v{APP_VERSION}", about_text)

    def _open_online_docs(self):
        """Opens the placeholder URL for online documentation."""
        # Replace with the actual URL when available
        url = QUrl("https://github.com/CosmicForge/libs-cosmic-forge") # Example URL
        logging.info(f"Attempting to open documentation URL: {url.toString()}")
        if not QDesktopServices.openUrl(url):
            logging.error(f"Could not open URL: {url.toString()}")
            QMessageBox.warning(self, "Cannot Open Link", f"Could not open the documentation link:\n{url.toString()}")

    def change_theme(self, theme_name: str):
        """Applies the selected theme and updates related settings."""
        logging.info(f"Changing theme to: {theme_name}")
        if theme_name not in self.theme_manager.get_available_themes():
             logging.warning(f"Attempted to switch to non-existent theme: {theme_name}")
             return

        success = self.theme_manager.apply_theme(theme_name)
        if success:
            self.config['default_theme'] = theme_name # Update config dict (might not persist unless saved elsewhere)
            self.update_theme_menu()
            self._apply_theme_to_plots()
            # Persist the theme change immediately using QSettings
            self._save_theme_setting(theme_name)
        else:
             logging.error(f"Theme manager failed to apply theme: {theme_name}")
             QMessageBox.warning(self, "Theme Error", f"Could not apply the theme '{theme_name}'. Check logs.")


    def _apply_theme_to_plots(self):
        """Applies current theme colors to relevant plot widgets."""
        logging.debug("Applying theme colors to plots.")
        plots_to_update = [
            self.plot_widget,
            self.boltzmann_view.boltzmann_plot_widget if self.boltzmann_view else None,
            self.ml_view.results_plot_widget if self.ml_view else None
            # Add other plot widgets here if they exist
        ]
        theme_applied_count = 0
        for plot in plots_to_update:
            # Check if plot exists, has the method, and is callable
            if plot and hasattr(plot, 'apply_theme_colors') and callable(plot.apply_theme_colors):
                try:
                    plot.apply_theme_colors(self.config) # Pass config or specific theme settings
                    theme_applied_count += 1
                except Exception as e:
                    logging.error(f"Error applying theme to plot {plot}: {e}", exc_info=True)
        logging.debug(f"Applied theme colors to {theme_applied_count} plot widgets.")


    def update_theme_menu(self):
        """Updates the check state of the theme actions in the menu."""
        current_theme = self.theme_manager.current_theme_name
        logging.debug(f"Updating theme menu. Current theme: {current_theme}")
        if not self.theme_actions:
             logging.warning("Theme actions dictionary is empty, cannot update menu.")
             return
        for name, action in self.theme_actions.items():
            is_checked = (name == current_theme)
            action.setChecked(is_checked)
            # logging.debug(f"Theme action '{name}': Checked={is_checked}")

    def closeEvent(self, event):
        """Handles the window close event."""
        logging.info("Close event triggered. Preparing to exit.")

        # 1. Ask user if they want to save unsaved changes (if applicable)
        #    (Implementation depends on how you track unsaved changes)
        # if self.has_unsaved_changes(): # Hypothetical method
        #    reply = QMessageBox.question(self, 'Exit Application',
        #                                 "There are unsaved changes. Save before exiting?",
        #                                 QMessageBox.StandardButton.Save |
        #                                 QMessageBox.StandardButton.Discard |
        #                                 QMessageBox.StandardButton.Cancel)
        #    if reply == QMessageBox.StandardButton.Save:
        #        if not self._on_save_session_triggered(): # Try saving, returns False if cancelled
        #            event.ignore() # User cancelled save, so cancel exit
        #            return
        #    elif reply == QMessageBox.StandardButton.Cancel:
        #        event.ignore() # Cancel the exit
        #        return
        #    # If Discard, proceed with closing

        # 2. Stop any running background tasks cleanly
        logging.debug("Stopping background tasks...")
        if self.nist_search_view and hasattr(self.nist_search_view, '_stop_running_search'):
             self.nist_search_view._stop_running_search()
        if self.external_process and self.external_process.state() != QProcess.ProcessState.NotRunning:
            logging.info("Terminating external process...")
            self.external_process.terminate() # Ask nicely first
            if not self.external_process.waitForFinished(1000): # Wait 1 sec
                logging.warning("External process did not terminate gracefully, killing.")
                self.external_process.kill() # Force kill if needed
            self.external_process = None # Clear reference

        # 3. Save settings if enabled
        self._save_persistent_settings()

        # 4. Accept the event to allow the window to close
        logging.info("Accepting close event. Application will exit.")
        event.accept()


    # --- Settings Persistence ---

    def _load_persistent_settings(self):
        """Loads window geometry, state, last directories, and theme using QSettings."""
        if not self.remember_window_state:
            logging.info("Window state persistence is disabled in config. Skipping load.")
            # Apply default theme if persistence is off
            default_theme_name = self.config.get('default_theme', DEFAULT_THEME)
            logging.info(f"Applying configured default theme: {default_theme_name}")
            self.theme_manager.apply_theme(default_theme_name)
            self._apply_theme_to_plots()
            self.update_theme_menu()
            # Use the already determined default directory
            # self._last_save_dir and self._last_load_dir were set in __init__
            logging.info(f"Using default load/save directory: {self._last_load_dir}")
            return

        logging.info("Loading persistent window settings using QSettings...")
        try:
            # Load geometry
            geometry_data = self.settings.value("MainWindow/geometry", defaultValue=None)
            if isinstance(geometry_data, QByteArray) and not geometry_data.isNull():
                if self.restoreGeometry(geometry_data):
                    logging.info("Restored window geometry.")
                else:
                    logging.warning("Failed to restore window geometry from settings (restoreGeometry returned False).")
            else:
                logging.info("No valid geometry found in settings, using default layout.")
                # self._setup_geometry() # Recalculate default based on screen if needed

            # Load window state (docks, toolbars)
            state_data = self.settings.value("MainWindow/windowState", defaultValue=None)
            if isinstance(state_data, QByteArray) and not state_data.isNull():
                 if self.restoreState(state_data):
                     logging.info("Restored window state (docks, toolbars).")
                 else:
                     logging.warning("Failed to restore window state from settings (restoreState returned False).")

            # Load last directories (use the already calculated default as fallback)
            default_dir = self._last_load_dir # Get the default determined in __init__
            self._last_save_dir = self.settings.value("Paths/lastSaveDir", defaultValue=default_dir)
            self._last_load_dir = self.settings.value("Paths/lastLoadDir", defaultValue=default_dir)
            logging.info(f"Restored last load directory: {self._last_load_dir}")
            logging.info(f"Restored last save directory: {self._last_save_dir}")

            # Load and apply last theme
            last_theme_name = self._load_last_theme() # This method handles defaults
            logging.info(f"Applying restored/default theme: {last_theme_name}")
            self.theme_manager.apply_theme(last_theme_name)
            self._apply_theme_to_plots() # Ensure plots get themed after load
            self.update_theme_menu() # Update menu check state

        except Exception as e:
            logging.error(f"Failed to load persistent window settings: {e}", exc_info=True)
            # Attempt to reset to sensible defaults on error
            self._setup_geometry()
            default_theme_name = self.config.get('default_theme', DEFAULT_THEME)
            self.theme_manager.apply_theme(default_theme_name)
            self._apply_theme_to_plots()
            self.update_theme_menu()


    def _save_persistent_settings(self):
        """Saves window geometry, state, last directories, and theme using QSettings."""
        if not self.remember_window_state:
            logging.info("Window state persistence is disabled. Skipping save.")
            return

        logging.info("Saving persistent window settings using QSettings...")
        try:
            # Use groups for better organization in QSettings
            self.settings.setValue("MainWindow/geometry", self.saveGeometry())
            self.settings.setValue("MainWindow/windowState", self.saveState())
            self.settings.setValue("Paths/lastSaveDir", self._last_save_dir)
            self.settings.setValue("Paths/lastLoadDir", self._last_load_dir)
            self.settings.setValue("Appearance/lastTheme", self.theme_manager.current_theme_name)

            # Sync forces writing to storage (important for reliability)
            self.settings.sync()
            status = self.settings.status()
            if status == QSettings.Status.NoError:
                 logging.info("Window settings saved successfully.")
            else:
                 logging.error(f"QSettings sync error while saving window settings: {status}")

        except Exception as e:
            logging.error(f"Failed to save persistent window settings: {e}", exc_info=True)


    def _save_theme_setting(self, theme_name: str):
         """Saves only the theme setting immediately using QSettings."""
         if not self.remember_window_state:
              # logging.debug("Theme persistence disabled, not saving theme change.")
              return
         try:
             self.settings.setValue("Appearance/lastTheme", theme_name)
             self.settings.sync()
             status = self.settings.status()
             if status == QSettings.Status.NoError:
                  logging.debug(f"Persisted theme setting: {theme_name}")
             else:
                  logging.error(f"QSettings sync error while saving theme: {status}")
         except Exception as e:
             logging.error(f"Failed to save theme setting immediately: {e}", exc_info=True)


    def _load_last_theme(self) -> str:
        """Loads the last used theme name from QSettings, falling back to config default."""
        config_default_theme = self.config.get('default_theme', DEFAULT_THEME)
        if not self.remember_window_state:
            return config_default_theme

        try:
            # Get saved theme, use config default if not found in settings
            last_theme = self.settings.value("Appearance/lastTheme", defaultValue=config_default_theme)

            # Validate if the loaded theme actually exists
            if last_theme in self.theme_manager.get_available_themes():
                logging.debug(f"Loaded last theme from settings: {last_theme}")
                return last_theme
            else:
                logging.warning(f"Saved theme '{last_theme}' not found or invalid. Using default '{config_default_theme}'.")
                # Optionally remove the invalid setting
                # self.settings.remove("Appearance/lastTheme")
                # self.settings.sync()
                return config_default_theme
        except Exception as e:
            logging.error(f"Failed to load last theme from settings: {e}. Using default '{config_default_theme}'.", exc_info=True)
            return config_default_theme


    # --- Action Handler / Slot Implementations ---

    @pyqtSlot()
    def load_spectrum_action(self):
        """Handles the 'Load Spectrum' action."""
        if self._is_busy:
            logging.warning("Load Spectrum action ignored while busy.")
            return

        logging.info("Triggered Load Spectrum action.")
        self.update_status("Opening file dialog...")
        file_filter = "Data Files (*.txt *.csv *.asc);;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Spectrum File",
            self._last_load_dir, # Use restored/default load directory
            file_filter
        )

        if filepath:
            self._last_load_dir = os.path.dirname(filepath) # Update for next time
            self.set_busy(True, f"Loading {os.path.basename(filepath)}...")
            spectrum = None
            try:
                # Use configured defaults for loading
                spectrum = load_spectrum_from_file(
                    filepath,
                    delimiter=self.default_delimiter,
                    comment_char=self.default_comment
                )
                # Reset state completely for the new single spectrum
                self._reset_state_for_new_spectrum(spectrum)
                # Explicitly enable relevant panels after successful load
                self._update_panel_enable_states(spectrum_loaded=True, peaks_detected=False, multi_loaded=False)

            except FileNotFoundError:
                self._handle_load_error(filepath, "File not found.")
            except (IOError, ValueError, IndexError, Exception) as e: # Catch common file reading/parsing errors
                self._handle_load_error(filepath, e)
            finally:
                self.set_busy(False)
        else:
            logging.info("Load Spectrum action cancelled by user.")
            self.update_status("Load cancelled.", 3000) # Temporary message

    @pyqtSlot()
    def _load_multiple_spectra_action(self):
        """Handles the 'Load Multiple Spectra' action."""
        if self._is_busy:
            logging.warning("Load Multiple Spectra action ignored while busy.")
            return

        logging.info("Triggered Load Multiple Spectra action.")
        self.update_status("Opening file dialog for multiple spectra...")
        file_filter = "Data Files (*.txt *.csv *.asc);;All Files (*)"
        filepaths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Multiple Spectra Files",
            self._last_load_dir,
            file_filter
        )

        if not filepaths:
            logging.info("Load Multiple Spectra action cancelled by user.")
            self.update_status("Load cancelled.", 3000)
            return

        self._last_load_dir = os.path.dirname(filepaths[0]) # Update based on first selected file
        self.set_busy(True, f"Loading {len(filepaths)} spectra...")

        loaded_spectra = []
        errors = []
        delimiter = self.config.get('file_io', {}).get('default_delimiter', '\t')
        comment = self.config.get('file_io', {}).get('default_comment_char', '#')

        try:
            for i, fp in enumerate(filepaths):
                # Provide progress in status bar
                self.update_status(f"Loading {i+1}/{len(filepaths)}: {os.path.basename(fp)}...", 0)
                QApplication.processEvents() # Keep UI responsive during loop
                try:
                    spectrum = load_spectrum_from_file(fp, delimiter=delimiter, comment=comment)
                    loaded_spectra.append(spectrum)
                except Exception as e:
                    error_msg = f"{os.path.basename(fp)}: {e}"
                    errors.append(error_msg)
                    logging.warning(f"Failed to load spectrum file {fp}: {e}")

            self.multi_spectra = loaded_spectra # Store the successfully loaded spectra
            if self.ml_view: self.ml_view.set_spectra_list(loaded_spectra) # Update ML view

            num_loaded = len(loaded_spectra)
            num_attempted = len(filepaths)
            status_msg = f"Loaded {num_loaded}/{num_attempted} spectra for ML."
            if errors: status_msg += " Some errors occurred."
            self.update_status(status_msg, 5000)


            if errors:
                # Show limited error details in message box
                error_summary = "\n".join([f"- {e}" for e in errors[:10]]) # Show up to 10 errors
                if len(errors) > 10:
                    error_summary += f"\n- ... ({len(errors)-10} more)"
                QMessageBox.warning(self, "Load Issues", f"Could not load all files:\n{error_summary}\n\nSee logs for full details.")

            if loaded_spectra:
                # Reset single spectrum state and update panel enables for ML mode
                self._reset_state_for_new_spectrum(None) # Clear single spectrum view/state
                self._update_panel_enable_states(spectrum_loaded=False, peaks_detected=False, multi_loaded=True)
                # Optionally switch focus to ML panel
                if 'ml_analysis' in self.docks:
                    self.docks['ml_analysis'].raise_() # Bring its tab to front
                    self.docks['ml_analysis'].show() # Ensure the dock itself is visible
            else:
                 # If no spectra loaded successfully
                 self._update_panel_enable_states(spectrum_loaded=False, peaks_detected=False, multi_loaded=False)

        except Exception as e:
            # Catch unexpected errors during the loading loop logic itself
            logging.error(f"Critical error during multi-spectrum load: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"An unexpected error occurred during loading:\n{e}")
            # Reset state thoroughly on critical failure
            self.multi_spectra = []
            if self.ml_view: self.ml_view.set_spectra_list([])
            self._reset_state_for_new_spectrum(None) # Also resets single spectrum state
            self._update_panel_enable_states(spectrum_loaded=False, peaks_detected=False, multi_loaded=False)
        finally:
            self.set_busy(False)


    @pyqtSlot()
    def _on_save_session_triggered(self) -> bool: # Return bool indicating success/cancel
        """Handles the 'Save Session' action. Returns True on success, False on cancel/fail."""
        if self._is_busy:
            logging.warning("Save Session action ignored while busy.")
            return False

        # Check if there's anything to save
        if not self.current_spectrum and not self.multi_spectra:
             QMessageBox.information(self, "Save Session", "No spectrum or analysis data loaded to save.")
             return False

        logging.info("Triggered Save Session action.")
        file_filter = f"LIBS Forge Session (*{SessionManager.SESSION_FILE_EXTENSION});;All Files (*)"

        # Suggest a filename based on current spectrum or default
        suggested_name = f"libs_forge_session{SessionManager.SESSION_FILE_EXTENSION}" # Default
        if self.current_spectrum and self.current_spectrum.filename:
            base_name = os.path.splitext(os.path.basename(self.current_spectrum.filename))[0]
            suggested_name = f"{base_name}{SessionManager.SESSION_FILE_EXTENSION}"

        default_path = os.path.join(self._last_save_dir, suggested_name)

        filepath, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Session As...",
            default_path,
            file_filter
        )

        if filepath:
            # Ensure correct extension (QFileDialog usually handles this based on filter, but double-check)
            if selected_filter.startswith("LIBS Forge Session") and not filepath.lower().endswith(SessionManager.SESSION_FILE_EXTENSION):
                filepath += SessionManager.SESSION_FILE_EXTENSION
                logging.debug(f"Appended session file extension: {filepath}")

            self._last_save_dir = os.path.dirname(filepath) # Update last used directory
            self.set_busy(True, f"Saving session to {os.path.basename(filepath)}...")
            save_successful = False
            try:
                # Delegate actual saving logic to SessionManager
                save_successful = self.session_manager.save_session(filepath)

                if save_successful:
                    self.update_status(f"Session saved: {os.path.basename(filepath)}", 5000)
                    logging.info(f"Session successfully saved to {filepath}")
                    # Mark changes as saved (if tracking unsaved changes)
                    # self.mark_changes_saved() # Hypothetical
                    return True
                else:
                    # save_session should ideally raise exceptions on failure,
                    # but handle boolean return just in case.
                    QMessageBox.warning(self, "Save Warning", "Session saving reported failure, but no specific error was raised. Check logs.")
                    self.update_status("Session save potentially failed.", 5000)
                    return False

            except (IOError, PermissionError, Exception) as e: # Catch specific and general errors
                logging.error(f"Failed to save session to {filepath}: {e}", exc_info=True)
                QMessageBox.critical(self, "Save Session Error", f"Could not save the session:\n{e}")
                self.update_status("Session save failed.", 5000)
                return False
            finally:
                self.set_busy(False)
        else:
            # User cancelled the dialog
            logging.info("Save Session action cancelled by user.")
            self.update_status("Save session cancelled.", 3000)
            return False # Indicate cancellation


    @pyqtSlot()
    def _on_load_session_triggered(self):
        """Handles the 'Load Session' action."""
        if self._is_busy:
            logging.warning("Load Session action ignored while busy.")
            return

        # --- Optional: Check for unsaved changes before loading ---
        # if self.has_unsaved_changes():
        #     reply = QMessageBox.question(self, 'Load Session',
        #                                  "Loading a session will discard current unsaved changes. Continue?",
        #                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        #     if reply == QMessageBox.StandardButton.No:
        #         self.update_status("Load session cancelled.", 3000)
        #         return
        # --- End Optional Check ---

        logging.info("Triggered Load Session action.")
        file_filter = f"LIBS Forge Session (*{SessionManager.SESSION_FILE_EXTENSION});;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Session",
            self._last_load_dir,
            file_filter
        )

        if filepath:
            self._last_load_dir = os.path.dirname(filepath) # Update last used directory
            self.set_busy(True, f"Loading session from {os.path.basename(filepath)}...")
            try:
                # Load the raw state dictionary first using SessionManager
                session_state = self.session_manager.load_session_data(filepath)

                if session_state:
                    # Apply the loaded state to the application
                    # This method handles resetting state and applying loaded data
                    self._apply_loaded_session_state(session_state)

                    self.update_status(f"Session loaded: {os.path.basename(filepath)}", 5000)
                    logging.info(f"Session successfully loaded from {filepath} and applied.")
                    # Mark changes as saved (since we just loaded)
                    # self.mark_changes_saved() # Hypothetical
                else:
                    # load_session_data returning None/empty might indicate a validation issue
                    QMessageBox.warning(self, "Load Session Warning", f"Session file loaded but contained no valid data or failed validation:\n{filepath}")
                    self.update_status("Session load failed (empty or invalid data).", 5000)
                    # Reset state to ensure clean slate after failed load
                    self._reset_state_for_new_spectrum(None)

            except FileNotFoundError:
                 logging.error(f"Session file not found: {filepath}")
                 QMessageBox.critical(self,"Load Session Error", f"Session file not found:\n{filepath}")
                 self.update_status("Session load failed (file not found).", 5000)
            except (ValueError, IOError, Exception) as e: # Catch file read/parse errors
                logging.error(f"Error reading or parsing session file {filepath}: {e}", exc_info=True)
                QMessageBox.critical(self, "Load Session Error", f"Error loading session file:\n{e}\n\nFile may be corrupted or incompatible.")
                self.update_status("Session load failed (read error).", 5000)
                # Reset state after load error
                self._reset_state_for_new_spectrum(None)
            finally:
                self.set_busy(False)
        else:
            # User cancelled the dialog
            logging.info("Load Session action cancelled by user.")
            self.update_status("Load session cancelled.", 3000)

    # --- State Management & Updates ---

    def _reset_state_for_new_spectrum(self, spectrum: Optional[Spectrum]):
        """
        Resets the application state, typically when loading a new single spectrum,
        loading multiple spectra, or clearing the state (pass None).
        Updates UI elements accordingly.
        """
        is_clearing_all = spectrum is None and not self.multi_spectra # Distinguish clearing vs loading multi
        logging.info(f"Resetting application state. New single spectrum: {'Yes' if spectrum else 'No'}. Multi-spectra mode: {'Yes' if self.multi_spectra else 'No'}.")

        # --- Clear Core Data ---
        self.current_spectrum = spectrum
        self.detected_peaks = []
        self.nist_matches = []
        self.plasma_temp_k = None
        self.electron_density_cm3 = None
        self.boltzmann_plot_data = None
        self.cf_libs_concentrations = None
        # Only clear multi_spectra if NOT loading multiple spectra
        if spectrum is not None or is_clearing_all:
             self.multi_spectra = []

        # --- Update UI Elements ---
        status_msg = "Ready."
        window_title = f"LIBS Forge v{APP_VERSION}"

        # Explicitly clear plot widget
        if self.plot_widget: self.plot_widget.clear_plot()

        if spectrum:
            # State for single spectrum loaded
            try:
                 status_msg = f"Loaded: {os.path.basename(spectrum.filename)} ({len(spectrum)} points)"
                 window_title = f"LIBS Forge - {os.path.basename(spectrum.filename)}"
                 logging.info(f"Spectrum loaded: {spectrum.filename}, {len(spectrum.wavelengths)} points.")
                 if self.plot_widget: self.plot_widget.plot_spectrum(spectrum) # Plot the new spectrum
            except AttributeError: # Handle case where spectrum object might be malformed
                 logging.error("Invalid Spectrum object passed to _reset_state_for_new_spectrum.")
                 status_msg = "Error: Invalid spectrum data."
                 self.current_spectrum = None # Nullify invalid spectrum
        elif self.multi_spectra:
             # State after loading multiple spectra (single spectrum view is cleared)
             status_msg = f"Ready for ML analysis ({len(self.multi_spectra)} spectra loaded)."
             logging.info("Application state reset for multi-spectra mode.")
        else:
             # State after clearing everything
             status_msg = "State cleared. No spectrum loaded."
             logging.info("Application state cleared.")


        self.update_status(status_msg)
        self.setWindowTitle(window_title)

        # --- Clear Downstream Analysis UI Elements ---
        if self.plot_widget:
            self.plot_widget.plot_peaks([]) # Clear peak markers
            self.plot_widget.clear_nist_matches() # Clear NIST markers
            self.plot_widget.plot_fit_lines([]) # Clear fit lines

        # Reset UI panel contents
        if self.peak_list_view: self.peak_list_view.update_peak_list([])
        if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(None) # Clear fit details
        if self.nist_search_view:
            self.nist_search_view.set_peaks_reference([]) # Clear peaks in NIST view
            self.nist_search_view.clear_results() # Clear NIST results table
        if self.boltzmann_view: self.boltzmann_view.clear_all()
        if self.cf_libs_view: self.cf_libs_view.clear_all()
        if self.ml_view:
            # Only clear ML view if NOT in multi-spectra mode
            if spectrum is not None or is_clearing_all:
                 self.ml_view.clear_all()
                 self.ml_view.set_spectra_list([]) # Clear ML view's list too

        # --- Update Panel Enabled States ---
        # This will be called separately after load actions complete

        # Ensure plot theme is reapplied after clearing/plotting
        self._apply_theme_to_plots()

        # Update save action enables based on the new state
        self._update_save_actions_state()

        # Mark changes as saved (since we just loaded/cleared)
        # self.mark_changes_saved() # Hypothetical


    def _handle_load_error(self, filepath: str, error: Any):
        """Handles errors during spectrum file loading and resets state."""
        error_str = str(error)
        logging.error(f"Failed to load spectrum from '{filepath}': {error_str}", exc_info=True)
        QMessageBox.critical(
            self,
            "Spectrum Load Error",
            f"Error loading file:\n{os.path.basename(filepath)}\n\nDetails:\n{error_str}"
            f"\n\nCheck file format, delimiter, and comment character settings."
        )
        self.update_status(f"Error loading {os.path.basename(filepath)}.", 5000)
        # Reset state thoroughly on load failure to avoid inconsistent state
        self._reset_state_for_new_spectrum(None)
        self._update_panel_enable_states(spectrum_loaded=False, peaks_detected=False, multi_loaded=False)


    def _apply_loaded_session_state(self, state: Dict[str, Any]):
        """
        Applies state loaded from a session file to the application.
        Handles potential errors during application.
        """
        logging.info("Applying loaded session state...")

        # 1. Reset current state to a clean slate BEFORE applying loaded state
        self._reset_state_for_new_spectrum(None) # Ensures UI elements are cleared

        try:
            # 2. Restore Window Geometry, State, and Theme first
            # (Using QSettings directly now, so these might be less critical in session file,
            # but keep for backward compatibility or if QSettings fails)
            if 'window_geometry' in state and state['window_geometry']:
                try:
                    geom_bytes = QByteArray.fromBase64(state['window_geometry'].encode('ascii'))
                    if not self.restoreGeometry(geom_bytes):
                        logging.warning("Session: Failed to restore window geometry.")
                except Exception as e: logging.error(f"Session: Error restoring geometry: {e}")
            if 'window_state' in state and state['window_state']:
                 try:
                     state_bytes = QByteArray.fromBase64(state['window_state'].encode('ascii'))
                     if not self.restoreState(state_bytes):
                         logging.warning("Session: Failed to restore window state (docks/toolbars).")
                 except Exception as e: logging.error(f"Session: Error restoring state: {e}")

            # Theme: Prioritize session theme, fallback to QSettings/default
            session_theme = state.get('current_theme')
            if session_theme and session_theme in self.theme_manager.get_available_themes():
                 self.change_theme(session_theme) # Applies, updates plots, saves setting
            else:
                 # If session theme is invalid/missing, load from QSettings/default
                 last_theme = self._load_last_theme()
                 self.change_theme(last_theme)

            # Restore last directories from session, fallback to current default
            default_dir = self._get_default_directory() # Recalculate default if needed
            self._last_load_dir = state.get('last_load_dir', default_dir)
            self._last_save_dir = state.get('last_save_dir', default_dir)
            logging.debug(f"Session restored load/save dirs: {self._last_load_dir} / {self._last_save_dir}")

            # 3. Restore Main Spectrum (Crucial Step)
            spectrum_path = state.get('current_spectrum_path')
            loaded_spectrum = None
            if spectrum_path:
                # Validate path exists before attempting load
                if not os.path.exists(spectrum_path):
                     logging.error(f"Session: Spectrum path '{spectrum_path}' not found.")
                     QMessageBox.critical(self, "Session Load Error", f"Spectrum file not found:\n{spectrum_path}\n\nCannot restore analysis state. Session load aborted.")
                     self._reset_state_for_new_spectrum(None) # Reset fully
                     return # Abort further state application

                # Get load parameters from session, fallback to current defaults
                delimiter = state.get('current_spectrum_delimiter', self.default_delimiter)
                comment = state.get('current_spectrum_comment', self.default_comment)
                try:
                    logging.info(f"Session: Reloading spectrum from {spectrum_path}")
                    loaded_spectrum = load_spectrum_from_file(spectrum_path, delimiter=delimiter, comment=comment)
                    # Assign to self.current_spectrum *after* successful load
                    self.current_spectrum = loaded_spectrum
                    if self.plot_widget: self.plot_widget.plot_spectrum(self.current_spectrum) # Plot it
                    self.setWindowTitle(f"LIBS Forge - {os.path.basename(self.current_spectrum.filename)}")
                    logging.info(f"Session restored spectrum: {self.current_spectrum.filename}")
                except Exception as e:
                    logging.error(f"Session: Failed to reload spectrum file '{spectrum_path}': {e}", exc_info=True)
                    QMessageBox.critical(self, "Session Load Error", f"Failed to reload the spectrum file referenced in the session:\n{spectrum_path}\n\nError: {e}\n\nSession load aborted.")
                    self._reset_state_for_new_spectrum(None) # Reset fully if core spectrum fails
                    return # Abort further state application
            else:
                 # No single spectrum path found - might be an ML session or incomplete save
                 logging.info("Session: No single spectrum path saved.")
                 # Proceed to load multi-spectra if present

            # 4. Restore Multi-Spectra List (Attempt Reload)
            # This part needs careful implementation. Reloading multiple files can be slow.
            # Option 1: Store paths and prompt user to reload if needed.
            # Option 2: Attempt background reloading (complex).
            # Option 3: Skip reloading here and let user reload manually. (Simplest)
            multi_paths = state.get('multi_spectra_paths', [])
            self.multi_spectra = [] # Ensure list is clear before attempting restore
            if multi_paths:
                 logging.info(f"Session contained {len(multi_paths)} multi-spectra paths. Attempting reload...")
                 # For now, just log and maybe update ML view with paths if it supports it
                 # Full reload might look like the _load_multiple_spectra_action logic
                 errors = []
                 delimiter = state.get('multi_spectra_delimiter', self.default_delimiter) # Use specific or fallback
                 comment = state.get('multi_spectra_comment', self.default_comment)
                 reloaded_multi = []
                 for i, fp in enumerate(multi_paths):
                     self.update_status(f"Session: Reloading multi-spectrum {i+1}/{len(multi_paths)}...", 0)
                     QApplication.processEvents()
                     if not os.path.exists(fp):
                          error_msg = f"File not found: {os.path.basename(fp)}"
                          errors.append(error_msg)
                          logging.warning(f"Session: Multi-spectrum path not found: {fp}")
                          continue
                     try:
                         spectrum = load_spectrum_from_file(fp, delimiter=delimiter, comment=comment)
                         reloaded_multi.append(spectrum)
                     except Exception as e:
                         error_msg = f"{os.path.basename(fp)}: {e}"
                         errors.append(error_msg)
                         logging.warning(f"Session: Failed to reload multi-spectrum file {fp}: {e}")

                 self.multi_spectra = reloaded_multi
                 if self.ml_view: self.ml_view.set_spectra_list(reloaded_multi)
                 if errors:
                     error_summary = "\n".join([f"- {e}" for e in errors[:5]])
                     if len(errors) > 5: error_summary += f"\n- ... ({len(errors)-5} more)"
                     QMessageBox.warning(self, "Session Load Warning", f"Could not reload all multi-spectra files referenced in session:\n{error_summary}")
                 logging.info(f"Session: Reloaded {len(reloaded_multi)}/{len(multi_paths)} multi-spectra.")

            # 5. Restore Panel Settings
            # Ensure panels exist before trying to set settings
            panel_keys = ['processing', 'detection', 'fitting', 'nist_search', 'boltzmann', 'cflibs', 'ml_analysis']
            for key in panel_keys:
                 settings = state.get(f"{key}_settings")
                 panel_widget = None
                 # Try finding the widget attribute robustly
                 if hasattr(self, f"{key}_panel"): panel_widget = getattr(self, f"{key}_panel")
                 elif hasattr(self, f"{key}_view"): panel_widget = getattr(self, f"{key}_view")

                 if settings and panel_widget:
                     # Check if the panel widget has the 'set_settings' method
                     if hasattr(panel_widget, 'set_settings') and callable(panel_widget.set_settings):
                         try:
                             panel_widget.set_settings(settings)
                             logging.debug(f"Session: Restored settings for panel: {key}")
                         except Exception as e:
                             logging.error(f"Session: Error restoring settings for panel '{key}': {e}", exc_info=True)
                     else:
                         logging.warning(f"Session: Panel/View '{key}' exists but has no 'set_settings' method.")
                 elif settings:
                     logging.warning(f"Session: Settings found for panel '{key}', but the panel widget itself was not found or initialized.")


            # 6. Restore Analysis Data (Peaks, Matches, Parameters etc.)
            # --- Peaks ---
            restored_peaks = []
            if 'detected_peaks' in state and self.current_spectrum: # Only restore peaks if single spectrum was restored
                 peak_data_list = state['detected_peaks']
                 logging.info(f"Session: Attempting to restore {len(peak_data_list)} peaks...")
                 # Check if Peak class has the required factory method
                 if hasattr(Peak, 'from_dict') and callable(Peak.from_dict):
                     valid_peaks = 0
                     for i, peak_data in enumerate(peak_data_list):
                         try:
                             p = Peak.from_dict(peak_data)
                             if p:
                                 restored_peaks.append(p)
                                 valid_peaks += 1
                             else: logging.warning(f"Session: Skipped invalid peak data (from_dict returned None) at index {i}: {peak_data}")
                         except (ValueError, TypeError, KeyError) as ve:
                             logging.warning(f"Session: Skipping invalid peak data during restore at index {i}: {peak_data}. Error: {ve}")
                     self.detected_peaks = restored_peaks
                     logging.info(f"Session: Successfully restored {valid_peaks}/{len(peak_data_list)} peak objects.")
                 else:
                     logging.error("Session: Peak class does not have 'from_dict' method. Cannot restore peaks.")
                     self.detected_peaks = []
            else:
                self.detected_peaks = [] # Ensure empty if no spectrum or no peak data

            # --- NIST Matches ---
            restored_matches = []
            if 'nist_matches' in state and state['nist_matches']:
                 match_data_list = state['nist_matches']
                 logging.info(f"Session: Attempting to restore {len(match_data_list)} NIST matches...")
                 if hasattr(NISTMatch, 'from_dict') and callable(NISTMatch.from_dict):
                     valid_matches = 0
                     for match_data in match_data_list:
                         try:
                             match = NISTMatch.from_dict(match_data) # Requires NISTMatch method
                             if match:
                                 restored_matches.append(match)
                                 valid_matches += 1
                         except Exception as e:
                              logging.warning(f"Session: Skipping invalid NIST match data: {match_data}. Error: {e}")
                     self.nist_matches = restored_matches
                     logging.info(f"Session: Successfully restored {valid_matches}/{len(match_data_list)} NIST Match objects.")
                     # Re-correlate restored matches to restored peaks
                     self._correlate_nist_matches_to_peaks()
                 else:
                     logging.error("Session: NISTMatch class does not have 'from_dict' method. Cannot restore matches.")
                     self.nist_matches = []
            else:
                 self.nist_matches = []


            # --- Plasma Parameters ---
            self.plasma_temp_k = state.get('plasma_temp_k')
            self.electron_density_cm3 = state.get('electron_density_cm3')
            logging.debug(f"Session: Restored T={self.plasma_temp_k} K, Ne={self.electron_density_cm3} cm⁻³")

            # --- Derived DataFrames (Boltzmann, CF-LIBS) ---
            # Needs careful handling of serialization format (e.g., JSON 'split' orient)
            self.boltzmann_plot_data = None
            if 'boltzmann_plot_data_json' in state and state['boltzmann_plot_data_json']:
                 try:
                      self.boltzmann_plot_data = pd.read_json(state['boltzmann_plot_data_json'], orient='split')
                      logging.debug("Session: Restored Boltzmann plot data DataFrame.")
                 except Exception as e:
                      logging.error(f"Session: Failed to restore Boltzmann plot data from JSON: {e}")

            self.cf_libs_concentrations = None
            if 'cf_libs_concentrations_json' in state and state['cf_libs_concentrations_json']:
                 try:
                      self.cf_libs_concentrations = pd.read_json(state['cf_libs_concentrations_json'], orient='split')
                      logging.debug("Session: Restored CF-LIBS concentrations DataFrame.")
                 except Exception as e:
                      logging.error(f"Session: Failed to restore CF-LIBS concentrations from JSON: {e}")


            # 7. Update UI with ALL Restored Data (after everything is loaded)
            logging.debug("Session: Updating UI elements with restored data...")
            if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
            if self.plot_widget:
                # Plot spectrum first (already done if loaded)
                # Then plot peaks, fits, and matches on top
                self.plot_widget.plot_peaks(self.detected_peaks)
                self.plot_widget.plot_fit_lines(self.detected_peaks) # Plot fits for restored peaks
                self.plot_widget.plot_nist_matches(self.nist_matches, correlate=False) # Plot restored & correlated matches
            if self.nist_search_view:
                 self.nist_search_view.set_peaks_reference(self.detected_peaks)
                 # Update NIST results table if method exists
                 if hasattr(self.nist_search_view, 'display_results') and self.nist_matches:
                      # Display results expects a DataFrame? Convert if necessary.
                      try:
                           if hasattr(NISTMatch, 'to_dict'):
                                nist_df = pd.DataFrame([m.to_dict() for m in self.nist_matches])
                                self.nist_search_view.display_results(nist_df)
                           else: logging.warning("Cannot update NIST results table: NISTMatch needs to_dict()")
                      except Exception as e:
                           logging.error(f"Session: Failed to update NIST view results table: {e}")

            # Update calculation views with restored data (if methods exist)
            if self.boltzmann_view and hasattr(self.boltzmann_view, 'set_restored_data'):
                 self.boltzmann_view.set_restored_data(self.plasma_temp_k, self.boltzmann_plot_data)
            if self.cf_libs_view and hasattr(self.cf_libs_view, 'set_restored_data'):
                 self.cf_libs_view.set_restored_data(self.plasma_temp_k, self.electron_density_cm3, self.cf_libs_concentrations)


            # 8. Update Panel Enable States based on final restored data
            self._update_panel_enable_states(
                 spectrum_loaded=(self.current_spectrum is not None),
                 peaks_detected=bool(self.detected_peaks),
                 multi_loaded=bool(self.multi_spectra)
            )

            # 9. Final UI Polish
            self._apply_theme_to_plots() # Ensure plots are themed correctly after all updates
            self._update_save_actions_state() # Update save enables based on restored data

            logging.info("Finished applying loaded session state.")

        except Exception as e:
             # Catch-all for unexpected errors during the state application process
             logging.error(f"Critical error applying loaded session state: {e}", exc_info=True)
             QMessageBox.critical(self, "Session Load Error", f"Failed to apply the loaded session state:\n{e}\n\nThe application state will be reset.")
             # Reset to a known clean state on critical failure during application
             self._reset_state_for_new_spectrum(None)
             self._update_panel_enable_states(spectrum_loaded=False, peaks_detected=False, multi_loaded=False)


    def _update_panel_enable_states(self, spectrum_loaded: bool, peaks_detected: bool, multi_loaded: bool):
        """Enables/disables dock widgets based on the current application state."""
        logging.debug(f"Updating panel states: SingleSpec={spectrum_loaded}, Peaks={peaks_detected}, MultiSpec={multi_loaded}")

        # Logic:
        # - Processing, Detection, Peak List: Need single spectrum loaded.
        # - Fitting, NIST, Boltzmann, CF-LIBS: Need single spectrum AND peaks detected/fitted.
        # - ML Analysis: Needs multiple spectra loaded.
        # These modes are mostly exclusive (loading multi clears single, loading single clears multi).

        enable_processing = spectrum_loaded and not multi_loaded
        enable_detection = spectrum_loaded and not multi_loaded
        enable_peak_list = spectrum_loaded and not multi_loaded
        enable_fitting = spectrum_loaded and peaks_detected and not multi_loaded
        enable_nist = spectrum_loaded and peaks_detected and not multi_loaded
        enable_boltzmann = spectrum_loaded and peaks_detected and not multi_loaded # Might add check for fits/matches later
        enable_cflibs = spectrum_loaded and peaks_detected and not multi_loaded # Might add check for temp/Ne later
        enable_ml = multi_loaded and not spectrum_loaded


        def set_panel_enabled(name: str, is_enabled: bool):
             """Safely enable/disable the widget inside a dock."""
             dock = self.docks.get(name)
             if dock and hasattr(dock, 'widget') and callable(dock.widget):
                 widget = dock.widget()
                 if widget:
                     widget.setEnabled(is_enabled)
                 else:
                     logging.warning(f"Dock '{name}' has no valid widget to enable/disable.")
             # else: logging.debug(f"Dock '{name}' not found for state update.") # Less verbose

        set_panel_enabled('processing', enable_processing)
        set_panel_enabled('detection', enable_detection)
        set_panel_enabled('peak_list', enable_peak_list)
        set_panel_enabled('fitting', enable_fitting)
        set_panel_enabled('nist_search', enable_nist)
        set_panel_enabled('boltzmann', enable_boltzmann)
        set_panel_enabled('cflibs', enable_cflibs)
        set_panel_enabled('ml_analysis', enable_ml)

        # Special case: Ensure plot widget interactions are enabled/disabled appropriately
        # (e.g., peak clicking might need peaks_detected)
        if self.plot_widget:
             # Example: Enable peak interaction only if peaks are present
             if hasattr(self.plot_widget, 'set_interaction_enabled'):
                  self.plot_widget.set_interaction_enabled(peaks_detected)


    # --- Core Processing/Analysis Slots ---

    @pyqtSlot(dict)
    def handle_process_request(self, settings: dict):
        """Handles baseline subtraction and smoothing requests."""
        if not self.current_spectrum:
            logging.warning("Process request ignored: No single spectrum loaded.")
            QMessageBox.warning(self, "Processing", "Please load a single spectrum before processing.")
            return
        if self._is_busy:
            logging.warning("Process request ignored: Application is busy.")
            return

        logging.info(f"Handling process request with settings: {settings}")
        self.set_busy(True, "Applying processing steps...")

        try:
            # --- Ensure we have raw data ---
            if self.current_spectrum.raw_intensity is None:
                 # If raw is missing, maybe copy processed if it exists? Risky. Best to fail.
                 logging.error("Cannot process: Current spectrum is missing raw_intensity data.")
                 QMessageBox.critical(self, "Processing Error", "Spectrum data is inconsistent (missing raw intensity). Cannot process.")
                 return # Exit the try block before finally

            wavelengths = self.current_spectrum.wavelengths
            # Work on a copy of the raw intensity to preserve it
            intensity = self.current_spectrum.raw_intensity.copy()
            baseline = np.zeros_like(intensity) # Initialize baseline as zeros

            # --- Baseline Correction ---
            baseline_method = settings.get('baseline_method', 'None')
            baseline_params = {k: v for k, v in settings.items() if k.startswith('baseline_')} # Filter params
            logging.debug(f"Baseline method: {baseline_method}, Params: {baseline_params}")

            if baseline_method == 'Polynomial':
                poly_order = settings.get('poly_order', 3) # Get poly order specifically if needed
                # baseline_poly might need different args, adjust call as needed
                intensity, baseline = baseline_poly(wavelengths, intensity, order=poly_order)
            elif baseline_method == 'SNIP':
                num_iterations = settings.get('num_iterations', 10) # Get SNIP iterations
                # baseline_snip might need different args, adjust call as needed
                intensity, baseline = baseline_snip(wavelengths, intensity, num_iterations=num_iterations)
            elif baseline_method == 'None':
                 logging.debug("No baseline subtraction applied.")
                 baseline = np.zeros_like(intensity) # Ensure baseline is zeros if none applied
            else:
                 logging.warning(f"Unknown baseline method requested: {baseline_method}. No baseline applied.")
                 baseline = np.zeros_like(intensity)


            # --- Smoothing (applied AFTER baseline removal) ---
            smoothing_method = settings.get('smoothing_method', 'None')
            smoothing_params = {k: v for k, v in settings.items() if k in ['window_length', 'polyorder']} # Filter SG params
            logging.debug(f"Smoothing method: {smoothing_method}, Params: {smoothing_params}")

            if smoothing_method == 'SavitzkyGolay':
                # Validate required params for SG
                wl = smoothing_params.get('window_length')
                po = smoothing_params.get('polyorder')
                if wl is not None and po is not None:
                    # Add validation: window must be odd, window > polyorder
                    if wl % 2 == 0:
                         logging.warning(f"Savitzky-Golay window length ({wl}) must be odd. Incrementing.")
                         wl += 1
                         smoothing_params['window_length'] = wl
                    if wl <= po:
                         logging.warning(f"Savitzky-Golay window length ({wl}) must be greater than polyorder ({po}). Skipping smoothing.")
                    else:
                        logging.debug(f"Applying Savitzky-Golay smoothing with params: {smoothing_params}")
                        intensity = smooth_savitzky_golay(intensity, **smoothing_params)
                else:
                    logging.warning("Savitzky-Golay smoothing requested but missing/invalid parameters (window_length, polyorder). Skipping.")
            elif smoothing_method == 'None':
                 logging.debug("No smoothing applied.")
            else:
                 logging.warning(f"Unknown smoothing method requested: {smoothing_method}. No smoothing applied.")

            # --- Update Spectrum and UI ---
            # Store the results back into the current_spectrum object
            self.current_spectrum.update_processed(intensity, baseline)
            logging.info("Processing complete. Spectrum object updated.")
            self.update_status("Processing complete.", 5000)

            # Update plot to show raw, processed, and baseline
            if self.plot_widget:
                self.plot_widget.plot_spectrum(
                    self.current_spectrum,
                    show_raw=True, # Show original raw data
                    show_processed=True, # Show the newly processed data
                    show_baseline=(baseline_method != 'None') # Show baseline if calculated
                )
                self._apply_theme_to_plots() # Reapply theme colors might be needed

            # --- Clear Downstream Analysis Results ---
            # Processing invalidates peaks, fits, NIST matches, and calculations
            logging.debug("Clearing downstream analysis results after processing.")
            self.detected_peaks = []
            self.nist_matches = []
            self.plasma_temp_k = None
            self.electron_density_cm3 = None
            self.boltzmann_plot_data = None
            self.cf_libs_concentrations = None

            # Clear UI elements displaying downstream data
            if self.plot_widget:
                self.plot_widget.plot_peaks([]) # Clear peaks from plot
                self.plot_widget.clear_nist_matches()
                self.plot_widget.plot_fit_lines([])
            if self.peak_list_view: self.peak_list_view.update_peak_list([])
            if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(None)
            if self.nist_search_view:
                self.nist_search_view.set_peaks_reference([])
                self.nist_search_view.clear_results()
            if self.boltzmann_view: self.boltzmann_view.clear_all()
            if self.cf_libs_view: self.cf_libs_view.clear_all()

            # Update panel enables (peaks are now gone, but processing is done)
            self._update_panel_enable_states(spectrum_loaded=True, peaks_detected=False, multi_loaded=False)
            # Update save action states (processed data is now available, others cleared)
            self._update_save_actions_state()

        except Exception as e:
            logging.error(f"Error during processing: {e}", exc_info=True)
            QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{e}")
            self.update_status("Processing failed.", 5000)
            # Optionally try to restore spectrum state if possible, or reset
            # self._reset_state_for_new_spectrum(self.current_spectrum) # Might replot original
        finally:
            self.set_busy(False) # Ensure busy state is cleared even on error


    @pyqtSlot(dict)
    def handle_peak_detection_request(self, settings: dict):
        """Handles peak detection requests."""
        # Check prerequisites
        if not self.current_spectrum:
             QMessageBox.warning(self, "Peak Detection", "Please load a single spectrum first.")
             return
        if self.current_spectrum.processed_intensity is None:
            QMessageBox.warning(self, "Peak Detection", "Spectrum has not been processed yet (e.g., baseline subtracted). Please run processing first.")
            return
        if self._is_busy:
            logging.warning("Peak detection request ignored: Busy.")
            return

        logging.info(f"Handling peak detection request with settings: {settings}")
        self.set_busy(True, "Detecting peaks...")

        try:
            # Make a local copy of settings to avoid modifying the original dict from the panel
            detection_settings = settings.copy()
            method = detection_settings.pop('method', 'Unknown') # Get method and remove from dict

            if method == 'ScipyFindPeaks':
                logging.debug(f"Using ScipyFindPeaks with params: {detection_settings}")
                # Ensure core function `detect_peaks_scipy` exists and is imported
                self.detected_peaks = detect_peaks_scipy(self.current_spectrum, **detection_settings)
            else:
                # Handle other methods if added later
                raise ValueError(f"Unsupported peak detection method selected: {method}")

            num_peaks = len(self.detected_peaks)
            logging.info(f"Peak detection complete. Found {num_peaks} peaks.")
            self.update_status(f"Found {num_peaks} peaks.", 5000)

            # Update UI with newly detected peaks
            if self.plot_widget:
                self.plot_widget.plot_peaks(self.detected_peaks) # Plot new peaks
                self._apply_theme_to_plots() # Reapply theme might be needed
                self.plot_widget.clear_nist_matches() # Clear old matches
                self.plot_widget.plot_fit_lines([]) # Clear old fits
            if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks) # Update list
            if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(None) # Clear fitting details view

            # Clear downstream analysis results (NIST, calculations) as peaks changed
            logging.debug("Clearing downstream analysis results after peak detection.")
            self.nist_matches = []
            self.plasma_temp_k = None
            self.electron_density_cm3 = None
            self.boltzmann_plot_data = None
            self.cf_libs_concentrations = None
            if self.nist_search_view:
                self.nist_search_view.set_peaks_reference(self.detected_peaks) # Update NIST view ref
                self.nist_search_view.clear_results() # Clear NIST table
            if self.boltzmann_view: self.boltzmann_view.clear_all()
            if self.cf_libs_view: self.cf_libs_view.clear_all()

            # Update panel enables (peaks are now available)
            self._update_panel_enable_states(spectrum_loaded=True, peaks_detected=num_peaks > 0, multi_loaded=False)
            # Update save actions (peak list is now available, others cleared)
            self._update_save_actions_state()

        except ValueError as ve: # Catch known errors like unsupported method
             logging.error(f"Peak detection configuration error: {ve}")
             QMessageBox.critical(self, "Peak Detection Error", str(ve))
             self.update_status("Peak detection failed (config error).", 5000)
             # Reset peak state on configuration failure
             self.detected_peaks = []
             if self.plot_widget: self.plot_widget.plot_peaks([])
             if self.peak_list_view: self.peak_list_view.update_peak_list([])
             if self.nist_search_view: self.nist_search_view.set_peaks_reference([])
             self._update_panel_enable_states(spectrum_loaded=True, peaks_detected=False, multi_loaded=False)
             self._update_save_actions_state()
        except Exception as e: # Catch unexpected errors in the detection function
            logging.error(f"Error during peak detection: {e}", exc_info=True)
            QMessageBox.critical(self, "Peak Detection Error", f"An unexpected error occurred during peak detection:\n{e}")
            self.update_status("Peak detection failed.", 5000)
            # Reset peak state on unexpected failure
            self.detected_peaks = []
            if self.plot_widget: self.plot_widget.plot_peaks([])
            if self.peak_list_view: self.peak_list_view.update_peak_list([])
            if self.nist_search_view: self.nist_search_view.set_peaks_reference([])
            self._update_panel_enable_states(spectrum_loaded=True, peaks_detected=False, multi_loaded=False)
            self._update_save_actions_state()
        finally:
            self.set_busy(False)


    @pyqtSlot(dict)
    def handle_peak_fitting_request(self, settings: dict):
        """Handles fitting all currently detected peaks."""
        # Check prerequisites
        if not self.detected_peaks:
            QMessageBox.warning(self, "Peak Fitting", "No peaks have been detected yet. Please run peak detection first.")
            return
        if not self.current_spectrum or self.current_spectrum.processed_intensity is None:
            # This check should ideally be covered by the fact that peaks only exist after processing
            QMessageBox.warning(self, "Peak Fitting", "Cannot fit peaks. Processed spectrum data is missing.")
            return
        if self._is_busy:
            logging.warning("Peak fitting request ignored: Busy.")
            return

        num_peaks_to_fit = len(self.detected_peaks)
        logging.info(f"Handling request to fit {num_peaks_to_fit} detected peaks with settings: {settings}")
        self.set_busy(True, f"Fitting {num_peaks_to_fit} peaks...")

        processed_intensity = self.current_spectrum.processed_intensity # Cache for efficiency in loop
        num_success = 0
        num_fail = 0

        try:
            # Make a local copy of settings to avoid modification issues if needed
            fitting_settings = settings.copy()
            fit_profile = fitting_settings.get('profile', 'Gaussian') # Get profile for logging

            for i, peak in enumerate(self.detected_peaks):
                # Update status periodically to show progress
                if i % max(1, (num_peaks_to_fit // 20)) == 0: # Update roughly 20 times or every peak if few
                    self.update_status(f"Fitting peak {i + 1}/{num_peaks_to_fit} ({fit_profile})...")
                    QApplication.processEvents() # Keep UI responsive

                try:
                    # Call the core fitting function (ensure it exists and is imported)
                    best_fit, all_fits = fit_peak(
                        spectrum=self.current_spectrum,
                        peak_index=peak.index, # Pass the index within the spectrum data array
                        processed_intensity=processed_intensity, # Pass the data to fit
                        **fitting_settings # Pass fitting parameters (profile, window, etc.)
                    )

                    # Store results back into the Peak object
                    peak.best_fit = best_fit
                    peak.alternative_fits = all_fits # Store alternative fits if needed later

                    if best_fit and best_fit.success:
                        num_success += 1
                        # Use detailed logging level for success details
                        logging.log(logging.DEBUG -1 if hasattr(logging, 'TRACE') else logging.DEBUG,
                                    f"Fit success: Peak {i} (Idx {peak.index}, Wl={peak.wavelength_detected:.3f}), Profile: {best_fit.profile_type}, Params: {best_fit.params}")
                    else:
                        num_fail += 1
                        reason = 'No best fit found' if not best_fit else best_fit.message
                        logging.warning(f"Fit failed: Peak {i} (Idx {peak.index}, Wl={peak.wavelength_detected:.3f}). Reason: {reason}")
                        peak.best_fit = None # Ensure failed fits are explicitly marked

                except Exception as e:
                    # Catch errors from the fit_peak function itself
                    logging.error(f"Error fitting peak {i} (Idx {peak.index}, Wl={peak.wavelength_detected:.3f}): {e}", exc_info=False) # exc_info=False keeps log cleaner for common fit errors
                    num_fail += 1
                    peak.best_fit = None
                    peak.alternative_fits = {}

            # --- Update UI after fitting all peaks ---
            logging.info(f"Bulk peak fitting complete. Success: {num_success}, Failed: {num_fail}")
            self.update_status(f"Fitting complete: {num_success}/{num_peaks_to_fit} successful.", 5000)

            # Update plot: replot peaks (markers might change) and add fit lines
            if self.plot_widget:
                self.plot_widget.plot_peaks(self.detected_peaks)
                self.plot_widget.plot_fit_lines(self.detected_peaks)
                self._apply_theme_to_plots() # Reapply theme
            # Update list view with fit information (e.g., fitted wavelength, intensity, profile)
            if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
            # Clear the detailed fitting panel view (as bulk fitting finished)
            if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(None)

            # --- Clear Downstream Analysis Results ---
            # Fits have changed, so Boltzmann, CF-LIBS, etc. are invalid
            logging.debug("Clearing downstream analysis results after peak fitting.")
            self.plasma_temp_k = None
            self.electron_density_cm3 = None
            self.boltzmann_plot_data = None
            self.cf_libs_concentrations = None
            # Clear UI for these results
            if self.boltzmann_view: self.boltzmann_view.clear_all()
            if self.cf_libs_view: self.cf_libs_view.clear_all()
            # NIST search results are potentially still valid if only fits changed,
            # but correlation might need update if fitted wavelengths shifted significantly.
            # Re-running correlation might be safest.
            self._correlate_nist_matches_to_peaks()
            if self.plot_widget: self.plot_widget.plot_nist_matches(self.nist_matches, correlate=True) # Replot possibly updated correlations


            # Update save action states (peak data now includes fits, others cleared/updated)
            self._update_save_actions_state()

        except Exception as e:
            # Catch unexpected errors during the fitting loop or setup
            logging.error(f"Critical error during bulk peak fitting: {e}", exc_info=True)
            QMessageBox.critical(self, "Peak Fitting Error", f"An unexpected error occurred during fitting:\n{e}")
            self.update_status("Peak fitting failed.", 5000)
            # Consider resetting peak fit state if needed
        finally:
            self.set_busy(False)


    @pyqtSlot(int, dict)
    def handle_refit_single_peak(self, peak_list_index: int, settings: dict):
        """Handles request to refit a single peak, typically from the PeakFitting panel."""
        # Check prerequisites
        if self._is_busy:
            logging.warning("Refit single peak ignored: Busy.")
            return
        if not (0 <= peak_list_index < len(self.detected_peaks)):
            logging.error(f"Invalid peak list index provided for refit: {peak_list_index}")
            QMessageBox.critical(self, "Refit Error", f"Internal error: Invalid peak index ({peak_list_index}).")
            return
        if not self.current_spectrum or self.current_spectrum.processed_intensity is None:
            logging.warning("Refit single peak ignored: No processed spectrum available.")
            QMessageBox.warning(self, "Refit Error", "Cannot refit peak: Processed spectrum data is missing.")
            return

        peak_to_refit = self.detected_peaks[peak_list_index]
        logging.info(f"Handling request to refit peak list index {peak_list_index} (Spectrum index {peak_to_refit.index}, Wl={peak_to_refit.wavelength_detected:.3f}) with settings: {settings}")
        self.set_busy(True, f"Refitting peak @ {peak_to_refit.wavelength_detected:.2f} nm...")

        processed_intensity = self.current_spectrum.processed_intensity
        original_fit = peak_to_refit.best_fit # Keep original for comparison/logging if needed
        best_fit = None # Initialize best_fit for UI update logic

        try:
            fitting_settings = settings.copy()
            # Call the core fitting function
            best_fit, all_fits = fit_peak(
                spectrum=self.current_spectrum,
                peak_index=peak_to_refit.index,
                processed_intensity=processed_intensity,
                **fitting_settings
            )
            # Update the specific peak object
            peak_to_refit.best_fit = best_fit
            peak_to_refit.alternative_fits = all_fits

            if best_fit and best_fit.success:
                logging.info(f"Refit successful for peak {peak_list_index}. Best profile: {best_fit.profile_type}")
                self.update_status(f"Refit successful for peak {peak_list_index}.", 3000)
            else:
                reason = 'No best fit found' if not best_fit else best_fit.message
                logging.warning(f"Refit failed for peak {peak_list_index}. Reason: {reason}")
                self.update_status(f"Refit failed for peak {peak_list_index}.", 3000)
                peak_to_refit.best_fit = None # Ensure failed fit is marked

            # Update UI elements
            # Refresh the peak list view to show updated fit parameters for this row
            if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
            # Update the fitting panel to show details of the *new* fit result
            if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(peak_to_refit)
            # Replot all peaks and fits, highlighting the one just refitted
            if self.plot_widget:
                self.plot_widget.plot_peaks(self.detected_peaks) # Update peak markers (in case pos changed slightly)
                self.plot_widget.plot_fit_lines(self.detected_peaks, highlight_fit=best_fit) # Redraw fits, highlight new one
                self._apply_theme_to_plots()

            # --- Clear/Update Downstream ---
            # Single refit *might* change Boltzmann/CF-LIBS. Safest to clear them.
            logging.debug("Clearing downstream analysis results after single peak refit.")
            self.plasma_temp_k = None
            self.electron_density_cm3 = None
            self.boltzmann_plot_data = None
            self.cf_libs_concentrations = None
            if self.boltzmann_view: self.boltzmann_view.clear_all()
            if self.cf_libs_view: self.cf_libs_view.clear_all()
            # NIST correlation might need update if fitted wavelength changed
            self._correlate_nist_matches_to_peaks()
            if self.plot_widget: self.plot_widget.plot_nist_matches(self.nist_matches, correlate=False)


            self._update_save_actions_state() # Peak data changed

        except Exception as e:
            logging.error(f"Error refitting peak {peak_list_index} (Index {peak_to_refit.index}): {e}", exc_info=True)
            QMessageBox.critical(self, "Refit Error", f"An error occurred while refitting the peak:\n{e}")
            self.update_status("Refit failed.", 5000)
            # Revert UI/state if desired, or just show the failed state
            peak_to_refit.best_fit = original_fit # Option: revert to original fit on error? Or keep as None?
            if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
            if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(peak_to_refit) # Show failed state

        finally:
            self.set_busy(False)


    @pyqtSlot(list) # Type hint for the list of NISTMatch objects
    def _handle_nist_search_results(self, matches: List[NISTMatch]):
        """Handles results received from the NIST search (online or local)."""
        # Check if busy state is needed here - NIST search might run in thread
        # self.set_busy(True, "Processing NIST results...") # Optional

        num_matches = len(matches)
        logging.info(f"Received {num_matches} potential NIST matches from search.")
        self.update_status(f"Received {num_matches} NIST matches.", 3000)

        self.nist_matches = matches # Store the raw match list

        # --- Correlate these matches to the currently detected peaks ---
        self._correlate_nist_matches_to_peaks()

        # --- Update UI ---
        # Update plot to show markers for correlated matches
        if self.plot_widget:
             self.plot_widget.plot_nist_matches(self.nist_matches, correlate=True) # Plot matches on spectrum
             self._apply_theme_to_plots()
        # Update the NIST search view's results table (if applicable and method exists)
        if self.nist_search_view and hasattr(self.nist_search_view, 'display_results'):
            # display_results might need a DataFrame, convert if necessary
            try:
                if hasattr(NISTMatch, 'to_dict'):
                     nist_df = pd.DataFrame([m.to_dict() for m in self.nist_matches])
                     self.nist_search_view.display_results(nist_df)
                else: logging.warning("Cannot update NIST results table: NISTMatch needs to_dict()")
            except Exception as e:
                logging.error(f"Failed to update NIST view results table: {e}")

        # --- Clear Downstream Analysis ---
        # New NIST matches invalidate previous Boltzmann/CF-LIBS calculations
        logging.debug("Clearing downstream analysis results after NIST search.")
        self.plasma_temp_k = None
        self.electron_density_cm3 = None
        self.boltzmann_plot_data = None
        self.cf_libs_concentrations = None
        if self.boltzmann_view: self.boltzmann_view.clear_all() # Boltzmann needs correlation + fits
        if self.cf_libs_view: self.cf_libs_view.clear_all() # CF-LIBS needs correlation + fits + temp/Ne

        # Update save state (NIST data is now available)
        self._update_save_actions_state()

        # self.set_busy(False) # Optional


    @pyqtSlot(str)
    def handle_boltzmann_populate_request(self, species: str):
        """Handles request from Boltzmann view to find candidate lines for a species."""
        if self._is_busy:
            logging.warning("Boltzmann populate request ignored: Busy.")
            return

        # --- Prerequisites Check ---
        if not self.boltzmann_view:
             logging.error("Boltzmann view not initialized, cannot populate.")
             return # Cannot proceed
        if not self.detected_peaks:
            self.update_status("Cannot populate Boltzmann lines: No peaks detected.", 3000)
            self.boltzmann_view.display_candidate_lines(pd.DataFrame()) # Clear table
            return
        # Check if peaks have fits (needed for intensity) and correlations (needed for atomic data)
        if not any(p.best_fit and p.best_fit.success for p in self.detected_peaks):
             self.update_status("Cannot populate Boltzmann lines: Peaks need successful fits.", 4000)
             self.boltzmann_view.display_candidate_lines(pd.DataFrame())
             return
        if not any(p.potential_matches for p in self.detected_peaks):
             self.update_status("Cannot populate Boltzmann lines: Run NIST search and correlation first.", 4000)
             self.boltzmann_view.display_candidate_lines(pd.DataFrame())
             return

        logging.info(f"Populating Boltzmann candidates for species: '{species}'")
        self.update_status(f"Finding candidate lines for {species}...")
        QApplication.processEvents()

        candidates = []
        try:
            # Validate species string format robustly
            species_clean = species.strip()
            parts = species_clean.split()
            if len(parts) != 2:
                 raise ValueError(f"Invalid species format: '{species}'. Expected 'Element IonState' (e.g., 'Fe I', 'Al II').")
            element_target, ion_state_target = parts[0].strip().lower(), parts[1].strip().lower()
            if not element_target or not ion_state_target:
                 raise ValueError(f"Invalid species format: '{species}'. Element and Ion State cannot be empty.")


            # Ensure correlation is up-to-date (optional, might be redundant if always run after NIST search)
            # self._correlate_nist_matches_to_peaks()

            # Define required atomic data keys for Boltzmann plot
            required_match_keys = ['ei', 'gi', 'aki'] # Upper state energy (E_k), stat weight (g_k), transition prob (A_ki)
            missing_data_count = 0

            for peak in self.detected_peaks:
                # Peak must have a successful fit for reliable intensity/area
                if not peak.best_fit or not peak.best_fit.success:
                    continue

                # Use fitted intensity or amplitude (check Peak object definition)
                # Assume peak.intensity_fitted holds the relevant value (e.g., amplitude or area)
                intensity_val = peak.intensity_fitted
                if intensity_val is None or not np.isfinite(intensity_val) or intensity_val <= 0:
                    logging.log(logging.DEBUG -1, f"Skipping peak {peak.index}: Invalid fitted intensity ({intensity_val}).")
                    continue # Skip peaks with invalid intensity

                # Check correlated matches for the target species
                found_candidate_for_peak = False
                for match in peak.potential_matches:
                    # Case-insensitive comparison is safer
                    match_element = match.element.strip().lower() if match.element else ""
                    match_ion_state = match.ion_state_str.strip().lower() if match.ion_state_str else ""

                    if (match_element == element_target and match_ion_state == ion_state_target):

                        # Check if essential data is present and valid numeric
                        missing_keys = []
                        valid_data = True
                        for key in required_match_keys:
                            val = getattr(match, key, None)
                            if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
                                missing_keys.append(key)
                                valid_data = False

                        if valid_data:
                            # All required data present and valid, add to candidates
                            candidates.append({
                                'Peak λ (nm)': peak.wavelength_fitted_or_detected, # Use best available wavelength
                                'Intensity': intensity_val,
                                'Elem': match.element,
                                'Ion': match.ion_state_str,
                                'DB λ (nm)': match.wavelength_db,
                                'E_k (eV)': match.ei, # Upper state energy
                                'g_k': match.gi, # Upper state statistical weight
                                'A_ki (s⁻¹)': match.aki, # Transition probability
                                'Peak Index': peak.index # Include peak index for reference/debugging
                            })
                            found_candidate_for_peak = True
                            # Found suitable match for this peak, move to next peak
                            break
                        else:
                             missing_data_count += 1
                             logging.log(logging.DEBUG -1, # Use detailed log level
                                f"Skipping match {match.element} {match.ion_state_str} ({match.wavelength_db:.3f} nm) for peak {peak.index} "
                                f"due to missing/invalid Boltzmann data: {missing_keys}.")

                # if not found_candidate_for_peak: # Log peaks skipped entirely?
                #     logging.log(logging.DEBUG -1, f"Peak {peak.index}: No matching {species} lines with valid data found.")


            # Create DataFrame from collected candidates
            candidate_df = pd.DataFrame(candidates)

            if not candidate_df.empty:
                 # Optional: Remove duplicates based on Peak Index if multiple matches per peak were possible
                 # (shouldn't happen with the 'break' above, but keep for safety)
                 initial_count = len(candidate_df)
                 candidate_df = candidate_df.drop_duplicates(subset=['Peak Index'])
                 final_count = len(candidate_df)
                 if initial_count != final_count:
                      logging.warning(f"Removed {initial_count - final_count} duplicate peak entries during Boltzmann population.")

                 logging.info(f"Found {final_count} unique candidate lines for {species}.")
                 if missing_data_count > 0:
                      logging.warning(f"Skipped {missing_data_count} potential line matches due to missing atomic data (E_k, g_k, A_ki).")
            else:
                logging.info(f"No suitable candidate lines found for {species} with required atomic data and successful fits.")
                if missing_data_count > 0:
                     logging.warning(f"Note: Skipped {missing_data_count} potential line matches due to missing atomic data.")


            # Display results in the Boltzmann view's table
            self.boltzmann_view.display_candidate_lines(candidate_df)
            self.update_status(f"Populated {len(candidate_df)} candidates for {species}.", 5000)

        except ValueError as ve:
             # Handle invalid species format
             logging.error(f"Invalid input for Boltzmann population: {ve}")
             QMessageBox.warning(self, "Boltzmann Plot Input Error", str(ve))
             if self.boltzmann_view: self.boltzmann_view.display_candidate_lines(pd.DataFrame()) # Clear table
             self.update_status("Boltzmann population failed (invalid input).", 5000)
        except Exception as e:
             # Catch unexpected errors
             logging.error(f"Error populating Boltzmann candidates for {species}: {e}", exc_info=True)
             QMessageBox.critical(self, "Boltzmann Plot Error", f"An error occurred while finding candidate lines:\n{e}")
             if self.boltzmann_view: self.boltzmann_view.display_candidate_lines(pd.DataFrame()) # Clear table on error
             self.update_status("Boltzmann population failed.", 5000)

    # Corrected Slot Signature - needs to match the signal from BoltzmannPlotView
    # Assuming signal is calculation_complete = pyqtSignal(bool, object, object, object)
    # -> success(bool), temperature(float/None), r_squared(float/None), plot_data(pd.DataFrame/None)
    @pyqtSlot(bool, object, object, object)
    def _handle_boltzmann_result(self, success: bool, temperature: Optional[float], r_squared: Optional[float], plot_data: Optional[pd.DataFrame]):
        """Stores the calculated plasma temperature and plot data from the Boltzmann plot."""
        logging.debug(f"Received Boltzmann result: success={success}, T={temperature}, R²={r_squared}, plot_data type={type(plot_data)}")

        # Validate received temperature and R²
        valid_temp = success and isinstance(temperature, (float, int)) and np.isfinite(temperature)
        valid_r2 = isinstance(r_squared, (float, int)) and np.isfinite(r_squared) if r_squared is not None else True # R² can be None/invalid even on success

        if valid_temp:
            self.plasma_temp_k = float(temperature)
            r2_str = f"(R²={r_squared:.4f})" if valid_r2 else "(R² invalid)"
            logging.info(f"Boltzmann calculation successful. Stored Plasma Temperature: {self.plasma_temp_k:.2f} K {r2_str}")
            self.update_status(f"Plasma Temperature calculated: {self.plasma_temp_k:.0f} K", 5000)

            # --- Update dependent components ---
            # Update CF-LIBS view with the new temperature
            if self.cf_libs_view: self.cf_libs_view.update_temperature(self.plasma_temp_k)
            # Clear Electron Density as it likely depends on temperature (if using Saha)
            self.electron_density_cm3 = None
            if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)
            # Clear Concentration results as they depend on Temp/Ne
            self.cf_libs_concentrations = None
            if self.cf_libs_view: self.cf_libs_view.display_concentrations(None) # Clear table

        else:
            # Calculation failed or returned invalid temperature
            self.plasma_temp_k = None
            logging.warning("Boltzmann calculation failed or returned invalid/non-finite temperature.")
            self.update_status("Plasma temperature calculation failed.", 5000)
            # Ensure dependent views and state are cleared
            if self.cf_libs_view:
                 self.cf_libs_view.update_temperature(None)
                 self.cf_libs_view.update_electron_density(None)
                 self.cf_libs_view.display_concentrations(None)
            self.electron_density_cm3 = None
            self.cf_libs_concentrations = None

        # --- Store the plot data (DataFrame) ---
        # Check if plot_data is indeed a pandas DataFrame
        if isinstance(plot_data, pd.DataFrame):
             self.boltzmann_plot_data = plot_data.copy() # Make a copy to be safe
             logging.debug(f"Stored Boltzmann plot data ({len(self.boltzmann_plot_data)} points).")
        else:
             self.boltzmann_plot_data = None # Clear previous data if new calculation failed to provide it
             if success: # Log warning only if calculation claimed success but gave no data
                 logging.warning("Could not retrieve valid plot data (DataFrame) from Boltzmann calculation signal, even though success=True.")

        # Update save action states (Boltzmann data might be available or cleared)
        self._update_save_actions_state()


    # Note: Implementation of Ne/Conc calculations depends heavily on `core.cflibs` module
    @pyqtSlot(str, str, float)
    def handle_ne_calculation_request(self, species1: str, species2: str, temperature_k: float):
        """Handles request to calculate electron density (Ne) using Saha-Boltzmann (Not Implemented)."""
        if self._is_busy:
            logging.warning("Electron density calculation request ignored: Busy.")
            return
        if not temperature_k or not np.isfinite(temperature_k):
            QMessageBox.warning(self, "Electron Density", "Cannot calculate Nₑ: Valid plasma temperature is required.")
            return
        if not self.detected_peaks or not any(p.best_fit and p.best_fit.success for p in self.detected_peaks):
             QMessageBox.warning(self, "Electron Density", "Cannot calculate Nₑ: Successfully fitted peaks are required.")
             return

        logging.info(f"Handling Nₑ calculation request for {species1}/{species2} at T={temperature_k:.0f} K.")
        self.set_busy(True, "Calculating Electron Density (Nₑ)...")
        QApplication.processEvents()

        ne_cm3 = None # Initialize
        try:
            # --- Placeholder Call ---
            # This requires finding appropriate lines for species1/species2, getting their atomic data,
            # using their fitted intensities, and applying the Saha-Boltzmann equation.
            # Requires access to partition functions (U) and ionization energies (E_ion).
            # Check if calculate_electron_density_saha exists and call it
            if callable(calculate_electron_density_saha):
                 # You would need to pass the relevant peaks, atomic data lookups etc.
                 # ne_cm3 = calculate_electron_density_saha(
                 #     self.detected_peaks, temperature_k, species1, species2,
                 #     self.partition_functions, self.ionization_energies # Example required data
                 # )
                 raise NotImplementedError("Saha-Boltzmann electron density calculation logic in 'calculate_electron_density_saha' is not implemented yet.")
            else:
                 raise NotImplementedError("Core function 'calculate_electron_density_saha' not found or not callable.")

            # --- If Calculation Succeeds (uncomment when implemented) ---
            # if ne_cm3 is not None and np.isfinite(ne_cm3):
            #      self.electron_density_cm3 = ne_cm3
            #      logging.info(f"Electron density calculated: {self.electron_density_cm3:.3e} cm⁻³")
            #      self.update_status(f"Nₑ calculated: {self.electron_density_cm3:.3e} cm⁻³", 5000)
            #      if self.cf_libs_view: self.cf_libs_view.update_electron_density(self.electron_density_cm3)
            # else:
            #      # Handle case where calculation runs but returns invalid result
            #      logging.warning(f"Nₑ calculation returned invalid result: {ne_cm3}")
            #      QMessageBox.warning(self,"Electron Density", "Calculation completed but resulted in an invalid value for Nₑ.")
            #      self.electron_density_cm3 = None
            #      if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)


        except NotImplementedError as nie:
             # Expected error if function is missing/not implemented
             logging.warning(f"Nₑ calculation prerequisite failed: {nie}")
             QMessageBox.warning(self, "Electron Density Calculation", str(nie))
             self.update_status(f"Nₑ calculation failed: Not implemented.", 5000)
             self.electron_density_cm3 = None
             if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)
        except ValueError as ve:
             # Catch errors related to finding lines, missing data etc. within the core function
             logging.warning(f"Nₑ calculation failed: {ve}")
             QMessageBox.warning(self, "Electron Density Calculation", f"Could not calculate Nₑ:\n{ve}")
             self.update_status(f"Nₑ calculation failed: {ve}", 5000)
             self.electron_density_cm3 = None
             if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)
        except Exception as e:
             # Catch unexpected errors during calculation
             logging.error(f"Error during electron density calculation: {e}", exc_info=True)
             QMessageBox.critical(self, "Calculation Error", f"An unexpected error occurred during Nₑ calculation:\n{e}")
             self.update_status("Nₑ calculation failed (unexpected error).", 5000)
             self.electron_density_cm3 = None
             if self.cf_libs_view: self.cf_libs_view.update_electron_density(None)
        finally:
            # --- Clear Concentration Results ---
            # Nₑ calculation attempt invalidates previous concentrations
            self.cf_libs_concentrations = None
            if self.cf_libs_view: self.cf_libs_view.display_concentrations(None) # Clear table
            self._update_save_actions_state() # Ne/Conc state changed
            self.set_busy(False)


    @pyqtSlot(float, object) # Accepts temperature(float), electron density (float/None)
    def handle_conc_calculation_request(self, temperature_k: float, ne_cm3_obj: Optional[object]):
        """Handles request to calculate concentrations using CF-LIBS (Not Implemented)."""
        if self._is_busy:
            logging.warning("Concentration calculation request ignored: Busy.")
            return

        # --- Validate Prerequisites ---
        if not self.detected_peaks or not any(p.best_fit and p.best_fit.success and p.potential_matches for p in self.detected_peaks):
            QMessageBox.warning(self, "CF-LIBS Calculation", "Cannot calculate concentrations. Fitted peaks with NIST correlations are required.")
            return
        if temperature_k is None or not np.isfinite(temperature_k):
             QMessageBox.warning(self, "CF-LIBS Calculation", "Cannot calculate concentrations. A valid plasma temperature (from Boltzmann plot) is required.")
             return
        # Validate Ne - allow None, but check for finite if provided
        ne_cm3 = None
        if ne_cm3_obj is not None:
             try:
                  ne_cm3 = float(ne_cm3_obj)
                  if not np.isfinite(ne_cm3):
                       QMessageBox.warning(self, "CF-LIBS Calculation", f"Invalid Electron Density provided ({ne_cm3_obj}). Please calculate or enter a valid Nₑ.")
                       return
             except (ValueError, TypeError):
                  QMessageBox.warning(self, "CF-LIBS Calculation", f"Invalid Electron Density provided ({ne_cm3_obj}). Must be a number.")
                  return


        logging.info(f"Handling concentration calculation request. T={temperature_k:.0f} K, Nₑ={f'{ne_cm3:.3e} cm⁻³' if ne_cm3 is not None else 'Not provided'}.")
        self.set_busy(True, "Calculating CF-LIBS Concentrations...")
        QApplication.processEvents()

        concentrations_df = None # Initialize
        try:
            # --- Placeholder Call ---
            # Requires iterating through fitted/correlated peaks, using T, Ne (if available),
            # partition functions (U), and potentially Saha equation to calculate C_s * F.
            # Then normalize results.
            if callable(calculate_cf_libs_conc):
                 # concentrations_df = calculate_cf_libs_conc(
                 #     self.detected_peaks, temperature_k, ne_cm3,
                 #     self.partition_functions # Example required data
                 # )
                 raise NotImplementedError("CF-LIBS concentration calculation logic in 'calculate_cf_libs_conc' is not implemented yet.")
            else:
                 raise NotImplementedError("Core function 'calculate_cf_libs_conc' not found or not callable.")


            # --- If Calculation Succeeds (uncomment when implemented) ---
            # if isinstance(concentrations_df, pd.DataFrame) and not concentrations_df.empty:
            #      self.cf_libs_concentrations = concentrations_df
            #      logging.info("CF-LIBS concentration calculation finished.")
            #      self.update_status("CF-LIBS concentrations calculated.", 5000)
            #      if self.cf_libs_view: self.cf_libs_view.display_concentrations(self.cf_libs_concentrations)
            # else:
            #      # Handle calculation running but returning empty/invalid result
            #      logging.warning(f"CF-LIBS calculation returned empty or invalid DataFrame: {concentrations_df}")
            #      QMessageBox.warning(self, "CF-LIBS Calculation", "Calculation completed but resulted in no valid concentration data.")
            #      self.cf_libs_concentrations = None
            #      if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)


        except NotImplementedError as nie:
             logging.warning(f"CF-LIBS calculation failed: {nie}")
             QMessageBox.warning(self, "CF-LIBS Calculation", str(nie))
             self.update_status(f"CF-LIBS calculation failed: Not implemented.", 5000)
             self.cf_libs_concentrations = None
             if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        except ValueError as ve:
             # Catch errors from core function (e.g., missing data, invalid inputs)
             logging.warning(f"CF-LIBS calculation failed: {ve}")
             QMessageBox.warning(self, "CF-LIBS Calculation", f"Could not calculate concentrations:\n{ve}")
             self.update_status(f"CF-LIBS calculation failed: {ve}", 5000)
             self.cf_libs_concentrations = None
             if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        except Exception as e:
             # Catch unexpected errors
             logging.error(f"Error during concentration calculation: {e}", exc_info=True)
             QMessageBox.critical(self, "Calculation Error", f"An unexpected error occurred during concentration calculation:\n{e}")
             self.update_status("Concentration calculation failed (unexpected error).", 5000)
             self.cf_libs_concentrations = None
             if self.cf_libs_view: self.cf_libs_view.display_concentrations(None)
        finally:
            # Update save state regardless of success/failure
            self._update_save_actions_state()
            self.set_busy(False)


    # --- UI Interaction Slots ---

    @pyqtSlot(int)
    def handle_peak_selection(self, peak_list_index: int):
        """Handles selection changes in the PeakListView."""
        if self._is_busy: return # Don't interfere with ongoing operations

        selected_peak: Optional[Peak] = None
        # Validate index against the current list of detected peaks
        if 0 <= peak_list_index < len(self.detected_peaks):
            selected_peak = self.detected_peaks[peak_list_index]
            logging.debug(f"Peak selected from list: Index {peak_list_index} (Wl={selected_peak.wavelength_fitted_or_detected:.3f})")
        else:
            # Handle deselection (index might be -1) or invalid index
            logging.debug(f"Peak selection cleared or invalid index from list: {peak_list_index}")

        # Update the fitting panel to show details of the selected peak (or None if deselected)
        if self.peak_fitting_panel:
             self.peak_fitting_panel.display_peak_fit_details(selected_peak)

        # Highlight the corresponding peak marker on the plot (or clear highlight)
        if self.plot_widget:
            # Pass index if selected, None otherwise
            self.plot_widget.highlight_peak(peak_list_index if selected_peak else None)
            # Highlight the fit line if available for the selected peak
            fit_to_highlight = selected_peak.best_fit if selected_peak and selected_peak.best_fit and selected_peak.best_fit.success else None
            self.plot_widget.highlight_fit_line(fit_to_highlight)


    @pyqtSlot(int)
    def handle_peak_plot_click(self, peak_plot_index: int):
        """Handles clicks on peak markers in the SpectrumPlotWidget."""
        if self._is_busy: return

        # peak_plot_index should correspond to the index in the list used for plotting (self.detected_peaks)
        if 0 <= peak_plot_index < len(self.detected_peaks):
            clicked_peak = self.detected_peaks[peak_plot_index]
            logging.info(f"Peak clicked on plot: Index {peak_plot_index} (Wl={clicked_peak.wavelength_fitted_or_detected:.3f})")

            # --- Action: Select the corresponding row in the PeakListView ---
            if self.peak_list_view:
                 # Check if the view has a method to select by index
                 if hasattr(self.peak_list_view, 'select_peak_by_index'):
                      self.peak_list_view.select_peak_by_index(peak_plot_index)
                      # The list view's selection change should trigger handle_peak_selection automatically
                 else:
                      logging.warning("PeakListView does not have 'select_peak_by_index' method.")
                      # Fallback: manually trigger update based on plot click? Less ideal.
                      # self.handle_peak_selection(peak_plot_index)
            else:
                 logging.warning("PeakListView not available to sync selection.")

        else:
            logging.warning(f"Invalid peak index received from plot click: {peak_plot_index}. List length: {len(self.detected_peaks)}")
            # Optional: Clear selection in list view if click was off-peak?
            # if self.peak_list_view: self.peak_list_view.clearSelection()


    @pyqtSlot(object) # Expecting FitResult object or None
    def handle_show_specific_fit(self, fit_result: Optional[FitResult]):
        """Highlights a specific fit line on the plot, usually requested by Fitting Panel."""
        if self._is_busy: return

        if fit_result:
            # Optional: Log details of the fit being highlighted
            logging.debug(f"Request from fitting panel to highlight fit: Profile={fit_result.profile_type}, RMSE={fit_result.rmse:.4f}")
        else:
            logging.debug("Request from fitting panel to clear fit highlight.")

        # Pass the FitResult object (or None) to the plot widget's highlight method
        if self.plot_widget:
             if hasattr(self.plot_widget, 'highlight_fit_line'):
                  self.plot_widget.highlight_fit_line(fit_result)
             else:
                  logging.warning("Plot widget does not have 'highlight_fit_line' method.")


    # --- Data Handling Helpers ---

    def _correlate_nist_matches_to_peaks(self):
        """Associates NISTMatch objects with the closest detected/fitted peaks within tolerance."""
        # Check if prerequisites are met
        if not self.detected_peaks:
            logging.debug("Correlation skipped: No peaks detected.")
            return # Nothing to correlate to
        if not self.nist_matches:
            logging.debug("Correlation skipped: No NIST matches available.")
            # Ensure potential_matches lists are cleared if NIST matches were removed
            for peak in self.detected_peaks:
                if peak.potential_matches: # Only clear if not already empty
                     peak.potential_matches.clear()
            # Update list view if needed to reflect removed correlations
            # if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
            return

        # Get tolerance from the NIST search view UI
        tolerance_nm = 0.1 # Default tolerance
        try:
            # Safely access the spinbox value
            if (self.nist_search_view and
                hasattr(self.nist_search_view, 'tolerance_spinbox') and
                self.nist_search_view.tolerance_spinbox is not None):
                 tolerance_nm = self.nist_search_view.tolerance_spinbox.value()
            else:
                 logging.warning("Could not get tolerance from NIST view UI (widget missing or None). Using default 0.1 nm.")
        except AttributeError:
            logging.warning("Attribute error accessing NIST tolerance spinbox (e.g., view not fully initialized). Using default 0.1 nm.")
        except Exception as e:
             logging.error(f"Unexpected error getting tolerance: {e}. Using default 0.1 nm.", exc_info=True)


        logging.info(f"Correlating {len(self.nist_matches)} NIST matches to {len(self.detected_peaks)} peaks with tolerance {tolerance_nm:.3f} nm.")

        # --- Step 1: Clear previous correlations from all peaks ---
        for peak in self.detected_peaks:
            peak.potential_matches.clear()

        # --- Step 2: Iterate through NIST matches and find the best peak match for each ---
        correlation_count = 0
        unmatched_nist_lines = 0

        for match in self.nist_matches:
            db_wavelength = match.wavelength_db
            # Skip NIST matches that lack a valid database wavelength
            if db_wavelength is None or not np.isfinite(db_wavelength):
                 logging.log(logging.DEBUG -1, f"Skipping NIST match due to invalid DB wavelength: {match}")
                 continue

            best_peak_match: Optional[Peak] = None
            min_diff = tolerance_nm # Max allowed difference

            # Find the *closest* peak within the tolerance window
            for peak in self.detected_peaks:
                # Use the best available wavelength for the peak (fitted preferred)
                peak_wavelength = peak.wavelength_fitted_or_detected
                if peak_wavelength is None or not np.isfinite(peak_wavelength):
                    continue # Skip peaks without a valid wavelength

                diff = abs(peak_wavelength - db_wavelength)

                # Check if within tolerance AND is closer than any previous candidate for *this* NIST line
                if diff <= min_diff:
                    min_diff = diff
                    best_peak_match = peak # Found a new best candidate peak

            # --- Step 3: Add the NIST match to the best-matching peak (if found) ---
            if best_peak_match:
                # Check if add_nist_match method exists and is callable
                if hasattr(best_peak_match, 'add_nist_match') and callable(best_peak_match.add_nist_match):
                    best_peak_match.add_nist_match(match)
                    correlation_count += 1
                    # Detailed log entry for successful correlation
                    logging.log(logging.DEBUG-1, # Trace level if available
                                f"Correlated NIST {match.element} {match.ion_state_str} @ {match.wavelength_db:.4f} nm "
                                f"to Peak {best_peak_match.index} @ {best_peak_match.wavelength_fitted_or_detected:.4f} nm "
                                f"(diff={min_diff:.4f} nm)")
                else:
                     logging.error(f"Peak object is missing the 'add_nist_match' method!")
                     # Abort or handle error? For now, just log.
            else:
                 # This NIST line did not match any peak within tolerance
                 unmatched_nist_lines += 1
                 logging.log(logging.DEBUG -1, # Trace level if available
                             f"NIST line {match.element} {match.ion_state_str} @ {match.wavelength_db:.4f} nm "
                             f"did not correlate to any peak within tolerance {tolerance_nm:.3f} nm.")


        logging.info(f"Correlation complete. Associated {correlation_count} NIST match instances to peaks. {unmatched_nist_lines} NIST lines remain uncorrelated.")

        # --- Step 4: Update UI that displays correlation info ---
        # Peak list view might have columns for correlated species/wavelengths
        if self.peak_list_view:
             # Assuming update_peak_list refreshes the display based on peak.potential_matches
             self.peak_list_view.update_peak_list(self.detected_peaks)


    def get_current_peaks(self) -> List[Peak]:
        """Public accessor for the current list of detected/processed peaks."""
        # Return a copy to prevent external modification? Or return reference?
        # Returning reference is usually fine for internal use.
        return self.detected_peaks


    # --- Save Actions ---

    # Renamed from update_save_actions_state to avoid conflict/confusion
    # Call this method whenever data that can be saved changes state (loaded, generated, cleared)
    # Already implemented earlier, ensure it's called _update_save_actions_state
    # def _update_save_actions_state(self): ...


    def _save_action(self, save_type: str):
        """Handles the dispatch for various 'Save...' actions."""
        if self._is_busy:
            logging.warning(f"Save action '{save_type}' ignored while busy.")
            return

        logging.info(f"Triggered save action for: {save_type}")

        # Determine filename base (use current spectrum or default)
        base_name = "libs_forge_output" # Generic default
        if self.current_spectrum and self.current_spectrum.filename:
            # Get filename without extension from the original loaded file
            base_name = os.path.splitext(os.path.basename(self.current_spectrum.filename))[0]

        # --- Configure save parameters based on type ---
        default_filename = ""
        file_filter = ""
        is_data_available = False # Flag to check if there's actually data to save
        save_function = None # Function to call for saving
        data_to_save = None # The actual data object/structure to pass
        save_kwargs = {} # Additional keyword args for the save function
        data_description = save_type.replace('_', ' ') # User-friendly description

        try:
            # --- Determine data and function based on save_type ---
            if save_type == 'processed_spectrum':
                data_description = "processed spectrum data"
                if self.current_spectrum and self.current_spectrum.processed_intensity is not None:
                    default_filename = f"{base_name}_processed.csv"
                    file_filter = "CSV Files (*.csv);;Text Files (*.txt)"
                    is_data_available = True
                    save_function = save_spectrum_data # Use the dedicated function
                    data_to_save = self.current_spectrum # Pass the whole spectrum object
                    # save_spectrum_data might need specific args, handled inside it
                # else: data is not available, is_data_available remains False

            elif save_type == 'peaks':
                data_description = "peak list"
                if self.detected_peaks: # Check if the list is not empty
                    default_filename = f"{base_name}_peaks.csv"
                    file_filter = "CSV Files (*.csv)"
                    is_data_available = True
                    save_function = save_peak_list # Use the dedicated function
                    data_to_save = self.detected_peaks # Pass the list of Peak objects
                # else: data is not available

            elif save_type == 'nist_matches':
                 data_description = "NIST match results"
                 nist_df_to_save = None
                 if self.nist_matches:
                     try:
                         # Requires NISTMatch objects to have a serialization method (e.g., to_dict)
                         if hasattr(NISTMatch, 'to_dict') and callable(NISTMatch.to_dict):
                              match_rows = [m.to_dict() for m in self.nist_matches]
                              nist_df_to_save = pd.DataFrame(match_rows)
                         else:
                             logging.warning("NISTMatch object missing 'to_dict' method, cannot save details to CSV.")
                             # Alternative: Save a simpler representation if possible
                     except Exception as e:
                         logging.error(f"Could not create DataFrame from nist_matches list: {e}")

                 # Check if DataFrame was created and is not empty
                 if nist_df_to_save is not None and not nist_df_to_save.empty:
                     default_filename = f"{base_name}_nist_matches.csv"
                     file_filter = "CSV Files (*.csv)"
                     is_data_available = True
                     save_function = save_dataframe # Use generic DataFrame saver
                     data_to_save = nist_df_to_save # Pass the DataFrame
                 # else: data is not available

            elif save_type == 'boltzmann':
                data_description = "Boltzmann plot data"
                # Check if the DataFrame exists and is not empty
                if self.boltzmann_plot_data is not None and not self.boltzmann_plot_data.empty:
                    default_filename = f"{base_name}_boltzmann_data.csv"
                    file_filter = "CSV Files (*.csv)"
                    is_data_available = True
                    save_function = save_dataframe # Use generic DataFrame saver
                    data_to_save = self.boltzmann_plot_data # Pass the DataFrame
                # else: data is not available

            elif save_type == 'concentrations':
                data_description = "CF-LIBS concentrations"
                # Check if the DataFrame exists and is not empty
                if self.cf_libs_concentrations is not None and not self.cf_libs_concentrations.empty:
                    default_filename = f"{base_name}_concentrations.csv"
                    file_filter = "CSV Files (*.csv)"
                    is_data_available = True
                    save_function = save_dataframe # Use generic DataFrame saver
                    data_to_save = self.cf_libs_concentrations # Pass the DataFrame
                # else: data is not available

            elif save_type == 'plot':
                 data_description = "main plot image"
                 # Check if plot widget exists and has data plotted
                 plot_is_valid = False
                 if self.plot_widget and hasattr(self.plot_widget, 'ax') and self.plot_widget.ax:
                      # Check if axes have any lines or collections drawn
                      if self.plot_widget.ax.lines or self.plot_widget.ax.collections:
                           plot_is_valid = True

                 if plot_is_valid:
                     default_filename = f"{base_name}_spectrum_plot.png" # Default to PNG
                     file_filter = "PNG Image (*.png);;SVG Vector (*.svg);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)"
                     is_data_available = True
                     # Special case: Saving the figure requires calling the figure's method
                     save_function = self.plot_widget.figure.savefig
                     # No data_to_save needed, function called directly with filepath
                     save_kwargs = {'dpi': 300, 'bbox_inches': 'tight'} # Sensible defaults for publication quality
                 # else: data is not available

            else:
                 # Should not happen if menu actions are correct
                 logging.error(f"Unknown save type requested: {save_type}")
                 QMessageBox.critical(self, "Save Error", f"Internal error: Unknown data type '{save_type}' requested for saving.")
                 return # Abort save

            # --- Show Save Dialog if data is available ---
            if not is_data_available:
                logging.warning(f"No data available to save for type: {save_type}")
                QMessageBox.information(self, "No Data to Save", f"There is no {data_description} available to save.")
                return # Abort save

            # Construct default path using the last used save directory
            default_path = os.path.join(self._last_save_dir, default_filename)

            # Use QFileDialog.getSaveFileName
            filepath, selected_filter = QFileDialog.getSaveFileName(
                self,
                f"Save {data_description.capitalize()} As...",
                default_path, # Suggest directory and filename
                file_filter # Provide file type options
            )

            # --- Execute Save if user provided a filepath ---
            if filepath:
                self._last_save_dir = os.path.dirname(filepath) # Update last used directory for next time
                self.set_busy(True, f"Saving {os.path.basename(filepath)}...")

                try:
                    if save_function:
                        # Special handling for savefig which takes filepath directly
                        if save_function == self.plot_widget.figure.savefig:
                             save_function(filepath, **save_kwargs) # Call fig.savefig(filepath, dpi=...)
                        else:
                             # Assume other functions follow the pattern: func(data, filepath=...)
                             save_function(data_to_save, filepath=filepath, **save_kwargs)

                        logging.info(f"{data_description.capitalize()} saved successfully to {filepath}")
                        self.update_status(f"Saved: {os.path.basename(filepath)}", 5000)
                    else:
                        # This case should have been caught earlier
                        raise NotImplementedError(f"Save function was not assigned for save type '{save_type}'.")

                except (IOError, PermissionError, Exception) as e: # Catch file errors and others
                    logging.error(f"Failed to save {save_type} to {filepath}: {e}", exc_info=True)
                    QMessageBox.critical(self, "Save Error", f"Failed to save {data_description}:\n{e}")
                    self.update_status("Save failed.", 5000)
                finally:
                    self.set_busy(False) # Ensure busy state is cleared

            else:
                # User cancelled the save dialog
                logging.info(f"Save action for '{save_type}' cancelled by user.")
                self.update_status("Save cancelled.", 3000)

        except Exception as e:
             # Catch errors during the setup phase of saving (before dialog)
             logging.error(f"Error preparing save action for '{save_type}': {e}", exc_info=True)
             QMessageBox.critical(self, "Save Error", f"An internal error occurred while preparing to save {data_description}:\n{e}")
             self.update_status("Save preparation failed.", 5000)
             if self._is_busy: self.set_busy(False) # Ensure busy state is off if error happened early


    # --- External Script Runner ---

    def run_external_script(self, script_relative_path: str, script_args: Optional[List[str]] = None):
        """Runs an external Python script using a dedicated dialog to show output."""
        if self._is_busy:
            logging.warning(f"Request to run script '{script_relative_path}' ignored while busy.")
            QMessageBox.warning(self, "Busy", "Cannot start an external script while another operation is in progress.")
            return

        script_args = script_args or [] # Ensure it's a list, even if empty

        try:
            project_root = get_project_root() # Get project root dynamically
            script_absolute_path = os.path.join(project_root, script_relative_path)
            script_absolute_path = os.path.normpath(script_absolute_path) # Normalize path separators

            # Check if the script file actually exists
            if not os.path.isfile(script_absolute_path):
                logging.error(f"External script not found at resolved path: {script_absolute_path}")
                QMessageBox.critical(self, "Script Not Found",
                                     f"The required script was not found:\n{script_absolute_path}\n"
                                     f"(Relative path: {script_relative_path}, Project Root: {project_root})")
                return

            logging.info(f"Preparing to run external script: {script_absolute_path} with args: {script_args}")

            # Use the dedicated dialog for running the script
            # Ensure ExternalScriptRunnerDialog is imported correctly
            dialog = ExternalScriptRunnerDialog(script_absolute_path, script_args, parent=self)
            dialog.exec() # Show the dialog modally (blocks until closed)

            logging.info(f"External script dialog closed for: {script_relative_path}. Exit code: {dialog.exit_code}")
            # You could check dialog.exit_code here if needed

        except ImportError:
             logging.error("Failed to import ExternalScriptRunnerDialog. Cannot run external scripts.", exc_info=True)
             QMessageBox.critical(self, "Dependency Error", "Could not find the ExternalScriptRunnerDialog component.")
        except Exception as e:
            logging.error(f"Error setting up or running external script runner for '{script_relative_path}': {e}", exc_info=True)
            QMessageBox.critical(self, "Script Runner Error", f"Could not initialize or run the script runner:\n{e}")

    @pyqtSlot()
    def run_nist_fetcher(self):
        """Runs the NIST data fetching script located in the 'database' directory."""
        # Relative path from project root
        self.run_external_script('database/nist_data_fetcher.py')

    @pyqtSlot()
    def run_atomic_data_builder(self):
        """Runs the atomic data building script (placeholder) located in 'database'."""
        # Relative path from project root
        # Add any specific arguments if the builder script needs them
        # e.g., self.run_external_script('database/atomic_data_builder.py', ['--output-dir', 'processed_data'])
        self.run_external_script('database/atomic_data_builder.py')

# Example usage (if this file were run directly for testing, though it's meant to be imported)
if __name__ == '__main__':
    # This block is for testing the MainWindow independently, if needed.
    # Usually, main.py would handle application setup.
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')

    # Minimal application setup for testing
    app = QApplication(sys.argv)

    # Dummy config for testing
    test_config = {
        'application': {'remember_window_state': True},
        'default_theme': 'dark_cosmic', # or another default
        'file_io': {'default_delimiter': '\t', 'default_comment_char': '#'},
        # Add other necessary config sections if required by panels/widgets
        'plotting': {'default_color': 'cyan', 'peak_marker_color': 'red'},
        'paths': {'nist_data': 'database/nist_data.db'} # Example path needed by a component
    }

    # Need a ThemeManager instance for testing themes if MainWindow depends on it
    # theme_manager = ThemeManager(app, test_config) # Might need creation before MainWindow if MainWindow uses it extensively in init

    main_win = MainWindow(test_config)
    main_win.show()

    sys.exit(app.exec())

