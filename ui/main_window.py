import logging
import os
import sys
import traceback
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable, Union
from pathlib import Path
import copy # For deep copy if needed

# --- Third-party Imports ---
import numpy as np
import pandas as pd
from PyQt6.QtCore import (
    QSize, Qt, pyqtSlot, QSettings, QByteArray, QPoint, QCoreApplication,
    QUrl, QProcess, QStandardPaths, QObject, QTimer # Added QTimer
)
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QCursor, QDesktopServices
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QStatusBar, QMenuBar,
    QMessageBox, QApplication, QFileDialog, QDockWidget, QToolBar, QMenu,
    QSizePolicy # Added QSizePolicy
)

# --- Application Metadata & Constants ---
# Moved to main.py to be set earlier
# APP_VERSION = "1.0.1"
# ORGANIZATION_NAME = "CosmicForgeDev"
# APPLICATION_NAME = "LIBSForge"
DEFAULT_THEME = "dark_cosmic"

# --- Logging Setup ---
# Assumes basic logging might be configured by main.py initially
log = logging.getLogger(__name__)

# --- Local Imports ---
# Use try-except for robustness, although main.py should handle critical failures
try:
    # Utilities first
    from utils.helpers import get_project_root
    from ui.theme import ThemeManager
    # Core components
    from core.data_models import Spectrum, Peak, NISTMatch, FitResult
    from core.file_io import (
        load_spectrum_from_file, save_spectrum_data, save_peak_list,
        save_nist_matches, save_dataframe
    )
    from core.session_manager import SessionManager
    from core.processing import baseline_poly, baseline_snip, smooth_savitzky_golay, denoise_wavelet
    from core.peak_detector import detect_peaks_scipy
    from core.peak_fitter import fit_peak, ProfileType, BaselineMode, ModelSelectionCriterion # Import enums
    from core.nist_manager import search_online_nist, ASTROQUERY_AVAILABLE # Ensure exists
    from core.cflibs import calculate_boltzmann_temp, calculate_electron_density_saha, calculate_cf_libs_conc # Ensure exists
    # UI Views and Widgets
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
except ImportError as e:
    # Log critical error if imports fail within MainWindow module
    log.critical(f"Failed to import necessary modules within MainWindow: {e}. Check installation and structure.", exc_info=True)
    # If QApplication exists, show message box, otherwise rely on console from main.py
    app = QApplication.instance()
    if app:
         QMessageBox.critical(None, "Initialization Error", f"Failed to import dependencies for MainWindow:\n{e}")
    # Re-raise to potentially stop execution if main.py didn't catch it somehow
    raise ImportError(f"Failed MainWindow imports: {e}") from e


# --- Enums for Clarity (Matching those in Panels/Core where applicable) ---
class DockName(Enum):
    PROCESSING = "processing"
    DETECTION = "detection"
    FITTING = "fitting"
    PEAK_LIST = "peak_list"
    NIST_SEARCH = "nist_search"
    BOLTZMANN = "boltzmann"
    CFLIBS = "cflibs"
    ML_ANALYSIS = "ml_analysis"

class PanelKey(Enum): # Used for session state keys
    PROCESSING = "processing_settings"
    DETECTION = "detection_settings"
    FITTING = "fitting_settings"
    NIST_SEARCH = "nist_search_settings"
    BOLTZMANN = "boltzmann_settings"
    CFLIBS = "cflibs_settings"
    ML_ANALYSIS = "ml_analysis_settings"
    # PEAK_LIST doesn't have settings currently, just displays data

class SaveType(Enum):
    SESSION = "session"
    PROCESSED_SPECTRUM = "processed_spectrum"
    PEAKS = "peaks"
    NIST_MATCHES = "nist_matches"
    BOLTZMANN_DATA = "boltzmann"
    CONCENTRATIONS = "concentrations"
    PLOT = "plot"

# --- Helper Function ---
def get_icon(name: str) -> QIcon:
    """Loads an icon from the assets folder or falls back to a theme icon."""
    try:
        project_root = Path(get_project_root()) # Use Path object
        icon_path = project_root / "assets" / "icons" / name
        if icon_path.is_file():
            return QIcon(str(icon_path))
        else:
            # Fallback: "load_spectrum.png" -> "document-open"
            theme_name = name.split('.')[0].replace('_', '-')
            fallback_icon = QIcon.fromTheme(theme_name)
            if fallback_icon.isNull():
                # Log only if theme icon AND file icon not found
                logging.debug(f"Icon '{name}' not found in assets path '{icon_path}' or theme '{theme_name}'.")
                return QIcon() # Return empty icon
            return fallback_icon
    except Exception as e:
         # Log error but return empty icon to prevent crashes
         log.error(f"Error getting icon '{name}': {e}. Returning empty icon.", exc_info=False)
         return QIcon()

# --- Main Window Class ---
class MainWindow(QMainWindow):
    """Main application window for LIBS Forge."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes the main window, UI components, and internal state."""
        super().__init__()
        log.info(f"Initializing LIBS Forge v{QCoreApplication.applicationVersion()}...")
        self.config = config
        # Set up application info early if not done in main.py
        self._setup_application_info() # Ensures QSettings uses correct paths
        self.settings = QSettings() # Load QSettings AFTER setting app info

        # --- Core Components ---
        self.theme_manager = ThemeManager(QApplication.instance(), self.config)
        self.session_manager = SessionManager(self)

        # --- Initialize UI Placeholders & App State ---
        self._init_ui_elements()
        self._init_app_state()
        self._load_config_defaults()
        self._init_ui_state()

        # --- Build UI ---
        # Wrap critical UI initialization in try/except
        try:
            self._init_ui() # Creates menus, toolbars, docks, central widget
            self._connect_signals()
            # Set initial enablement *after* UI is created
            self._update_action_panel_states()
            # Load persistent geometry, theme etc. *after* UI elements exist
            self._load_persistent_settings()
        except Exception as e_init:
             log.critical(f"CRITICAL ERROR during MainWindow UI initialization: {e_init}", exc_info=True)
             QMessageBox.critical(self, "Initialization Error", f"Failed to initialize main window UI components:\n{e_init}")
             # Consider exiting or disabling features if core UI fails
             # For now, allow potentially partial UI to show, but log critically.
             # If central widget failed, self.plot_widget might be None.

        # Final setup
        self.update_status(f"Welcome to LIBS Forge v{QCoreApplication.applicationVersion()}!")
        # Ensure final state update after loading settings
        self._update_ui_from_current_state() # Use the master update function
        log.info(f"LIBS Forge v{QCoreApplication.applicationVersion()} initialization complete.")

    def _setup_application_info(self):
        """Sets application name and organization for QSettings if not already set."""
        # Check if already set (e.g., by main.py)
        if not QCoreApplication.organizationName():
             QCoreApplication.setOrganizationName("CosmicForgeDev") # Use constant/config
        if not QCoreApplication.applicationName():
             QCoreApplication.setApplicationName("LIBSForge") # Use constant/config
        # Version set by main.py typically
        self.setWindowTitle(f"{QCoreApplication.applicationName()} v{QCoreApplication.applicationVersion()}")

    def _init_ui_elements(self):
        """Initializes UI element attributes to None."""
        self.plot_widget: Optional[SpectrumPlotWidget] = None
        self.processing_panel: Optional[ProcessingControlPanel] = None
        self.peak_detection_panel: Optional[PeakDetectionControlPanel] = None
        self.peak_fitting_panel: Optional[PeakFittingControlPanel] = None
        self.peak_list_view: Optional[PeakListView] = None
        self.nist_search_view: Optional[NistSearchView] = None
        self.boltzmann_view: Optional[BoltzmannPlotView] = None
        self.cf_libs_view: Optional[CfLibsView] = None
        self.ml_view: Optional[MLAnalysisView] = None
        self.docks: Dict[DockName, QDockWidget] = {}
        self.status_label: Optional[QLabel] = None
        self.menubar: Optional[QMenuBar] = None
        self.panels_menu: Optional[QMenu] = None
        self.save_menu: Optional[QMenu] = None
        self.theme_actions: Dict[str, QAction] = {}
        self.main_toolbar: Optional[QToolBar] = None
        # --- Action Attributes ---
        self.load_action: Optional[QAction] = None
        self.load_multi_action: Optional[QAction] = None
        self.load_session_action: Optional[QAction] = None
        self.save_session_action: Optional[QAction] = None
        self.save_processed_action: Optional[QAction] = None
        self.save_peaks_action: Optional[QAction] = None
        self.save_nist_action: Optional[QAction] = None
        self.save_boltzmann_action: Optional[QAction] = None
        self.save_conc_action: Optional[QAction] = None
        self.save_plot_action: Optional[QAction] = None
        self.save_plot_toolbar_action: Optional[QAction] = None
        self.reset_zoom_action: Optional[QAction] = None
        # Store other actions if they need to be enabled/disabled
        self.file_menu_actions: List[QAction] = []
        self.view_menu_actions: List[QAction] = []
        self.tools_menu_actions: List[QAction] = []
        self.toolbar_actions: List[QAction] = []


    def _init_app_state(self):
        """Initializes core application data state attributes."""
        self.current_spectrum: Optional[Spectrum] = None
        self.multi_spectra: List[Spectrum] = []
        self.detected_peaks: List[Peak] = []
        self.nist_matches: List[NISTMatch] = [] # Raw matches from search
        self.plasma_temp_k: Optional[float] = None
        self.electron_density_cm3: Optional[float] = None
        self.boltzmann_plot_data: Optional[pd.DataFrame] = None
        self.cf_libs_concentrations: Optional[pd.DataFrame] = None
        # State flags
        self._is_busy: bool = False
        self._is_dirty: bool = False # Track unsaved changes

    def _load_config_defaults(self):
        """Loads default settings from the configuration dictionary."""
        # Primarily for settings used directly by MainWindow
        file_io_cfg = self.config.get('file_io', {})
        app_cfg = self.config.get('application', {})
        self.default_delimiter = file_io_cfg.get('default_delimiter', '\t')
        self.default_comment_char = file_io_cfg.get('default_comment_char', '#')
        self.remember_window_state = app_cfg.get('remember_window_state', True)
        # Other defaults are loaded directly into panels

    def _init_ui_state(self):
        """Initializes UI-related state variables (like last used paths)."""
        try:
            default_dir = self._get_default_directory()
        except Exception as e:
            log.error(f"Error getting default directory: {e}", exc_info=True)
            default_dir = str(Path.home()) # Use pathlib for home dir
        self._last_save_dir: str = default_dir
        self._last_load_dir: str = default_dir
        self.external_process: Optional[QProcess] = None # For external script runner if needed

    def _get_default_directory(self) -> str:
        """Returns the user's default documents or home directory."""
        try:
            # Use DocumentsLocation first
            docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
            if docs and os.path.isdir(docs):
                log.debug(f"Using default directory: {docs}")
                return docs
            # Fallback to HomeLocation
            home = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation)
            if home and os.path.isdir(home):
                log.debug(f"DocumentsLocation not found or invalid, using home directory: {home}")
                return home
            # Final fallback using pathlib
            fallback = str(Path.home())
            log.warning(f"Could not find standard Documents or Home locations. Using fallback: {fallback}")
            return fallback
        except Exception as e:
             log.error(f"Error getting standard paths: {e}. Falling back to home dir.", exc_info=True)
             return str(Path.home())

    # --- UI Initialization Methods (_init_ui calls these) ---

    def _init_ui(self):
        """Initializes the main UI layout, widgets, menus, toolbars, and docks."""
        self.setWindowTitle(f"{QCoreApplication.applicationName()} v{QCoreApplication.applicationVersion()}")
        self._setup_geometry() # Set initial size based on screen or default
        self._setup_icon()
        self._setup_status_bar()
        # Central widget creation needs robustness
        self._setup_central_widget()
        # Create menus, toolbar, docks AFTER central widget (in case toolbar needs plot actions)
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_dock_widgets()
        self.setDockNestingEnabled(True)
        # Set initial enable/disable states AFTER all UI elements are created
        self._update_action_panel_states()

    def _setup_geometry(self):
        """Sets the initial window size and position based on screen."""
        # Check if geometry will be restored by settings later
        if self.remember_window_state and self.settings.contains("MainWindow/geometry"):
             log.debug("Window geometry will be restored from settings.")
             # Set a reasonable minimum size in case saved geometry is too small
             self.setMinimumSize(800, 600)
             # restoreGeometry() will be called in _load_persistent_settings
             return

        # If not restoring, set default size based on screen
        try:
            screen = QApplication.primaryScreen()
            if not screen:
                log.warning("Could not get primary screen. Using default size 1400x900.")
                self.resize(1400, 900)
                return

            available_geo = screen.availableGeometry()
            # Calculate default size as a percentage of available screen space
            width = int(available_geo.width() * 0.85)
            height = int(available_geo.height() * 0.85)
            # Ensure minimum size
            width = max(width, 1000)
            height = max(height, 700)
            # Center the window
            x_pos = int(available_geo.left() + (available_geo.width() - width) / 2)
            y_pos = int(available_geo.top() + (available_geo.height() - height) / 2)

            self.setGeometry(x_pos, y_pos, width, height)
            self.setMinimumSize(800, 600) # Set minimum size constraint
            log.debug(f"Initial geometry set based on screen: Pos=({x_pos},{y_pos}), Size=({width}x{height})")
        except Exception as e:
            log.warning(f"Could not determine screen geometry, using default size 1400x900. Error: {e}", exc_info=False)
            self.resize(1400, 900)
            self.setMinimumSize(800, 600)

    def _setup_icon(self):
        """Sets the application window icon."""
        app_icon = get_icon("app_icon.png") # Provide fallback?
        if not app_icon.isNull():
             self.setWindowIcon(app_icon)
        else:
             log.warning("Application icon 'app_icon.png' not found.")

    def _setup_status_bar(self):
        """Creates and configures the status bar."""
        statusBar = QStatusBar(self)
        self.setStatusBar(statusBar)
        self.status_label = QLabel("Initializing...") # Permanent label on the right
        self.status_label.setObjectName("StatusLabel") # For potential styling
        statusBar.addPermanentWidget(self.status_label)

    def _setup_central_widget(self):
        """Creates the central plot widget."""
        # Check if plot widget creation might fail and handle it
        try:
            self.plot_widget = SpectrumPlotWidget(parent=self, config=self.config)
            self.setCentralWidget(self.plot_widget)
            log.info("Central plot widget created successfully.")
        except ImportError as e_plot_deps:
             log.critical(f"Failed to create plot widget due to missing core dependencies: {e_plot_deps}. Plotting disabled.", exc_info=True)
             # Show a placeholder widget instead
             from ui.views.placeholder_view import PlaceholderView # Import locally
             placeholder = PlaceholderView("Plotting disabled: Core component import failed.\nCheck logs for details.")
             self.setCentralWidget(placeholder)
             self.plot_widget = None # Ensure plot_widget is None
        except Exception as e_plot_init:
            log.critical(f"Failed to create central plot widget: {e_plot_init}", exc_info=True)
            # Show a placeholder widget if creation fails for other reasons
            from ui.views.placeholder_view import PlaceholderView
            placeholder = PlaceholderView(f"Error: Plot widget failed to load.\nDetails: {e_plot_init}")
            self.setCentralWidget(placeholder)
            self.plot_widget = None # Ensure plot_widget is None


    def _create_menu_bar(self):
        """Creates the main menu bar and its submenus."""
        self.menubar = self.menuBar()
        if not self.menubar:
            log.error("Could not get QMenuBar instance.")
            return # Cannot proceed without menubar
        self._create_file_menu(self.menubar)
        self._create_view_menu(self.menubar)
        self._create_tools_menu(self.menubar)
        self._create_help_menu(self.menubar)

    def _create_file_menu(self, menubar: QMenuBar):
        """Creates the File menu and its actions."""
        file_menu = menubar.addMenu("&File")
        self.file_menu_actions = [] # Store actions for later enable/disable

        # --- Load Actions ---
        self.load_action = self._create_action( "Load &Spectrum...", "document-open", "Load a single spectrum file", self.load_spectrum_action, shortcut=QKeySequence.StandardKey.Open)
        file_menu.addAction(self.load_action)
        self.file_menu_actions.append(self.load_action)

        self.load_multi_action = self._create_action( "Load &Multiple Spectra...", "document-open-multiple", "Load multiple spectra for ML analysis", self._load_multiple_spectra_action)
        file_menu.addAction(self.load_multi_action)
        self.file_menu_actions.append(self.load_multi_action)

        self.load_session_action = self._create_action( "Load Session...", "document-open", "Load a previously saved analysis session", self._on_load_session_triggered, shortcut="Ctrl+L")
        file_menu.addAction(self.load_session_action)
        self.file_menu_actions.append(self.load_session_action)

        file_menu.addSeparator()

        # --- Save Menu ---
        self.save_menu = file_menu.addMenu(get_icon("document-save"), "&Save")

        self.save_session_action = self._create_action( "Save Session...", "document-save-as", "Save the current analysis state", self._on_save_session_triggered, shortcut=QKeySequence.StandardKey.Save, initial_enabled=False)
        self.save_menu.addAction(self.save_session_action)
        self.save_menu.addSeparator()

        # Individual Save Actions (use enum for type safety)
        self.save_processed_action = self._create_action("Processed Spectrum (.csv)", None, "Save processed data", lambda: self._save_action(SaveType.PROCESSED_SPECTRUM), False)
        self.save_menu.addAction(self.save_processed_action)
        self.save_peaks_action = self._create_action("Peak List (.csv)", None, "Save detected/fitted peaks", lambda: self._save_action(SaveType.PEAKS), False)
        self.save_menu.addAction(self.save_peaks_action)
        self.save_nist_action = self._create_action("NIST Matches (.csv)", None, "Save NIST match results", lambda: self._save_action(SaveType.NIST_MATCHES), False)
        self.save_menu.addAction(self.save_nist_action)
        self.save_boltzmann_action = self._create_action("Boltzmann Data (.csv)", None, "Save Boltzmann plot points", lambda: self._save_action(SaveType.BOLTZMANN_DATA), False)
        self.save_menu.addAction(self.save_boltzmann_action)
        self.save_conc_action = self._create_action("Concentrations (.csv)", None, "Save calculated concentrations", lambda: self._save_action(SaveType.CONCENTRATIONS), False)
        self.save_menu.addAction(self.save_conc_action)
        self.save_plot_action = self._create_action("Plot Image (.png, .svg)...", "image-x-generic", "Save plot as image", lambda: self._save_action(SaveType.PLOT), False)
        self.save_menu.addAction(self.save_plot_action)

        # Add Save menu actions to list for enable/disable
        self.file_menu_actions.extend(self.save_menu.actions())

        file_menu.addSeparator()

        # --- Exit Action ---
        exit_action = self._create_action( "E&xit", "application-exit", "Exit the application", self.close, shortcut=QKeySequence.StandardKey.Quit)
        file_menu.addAction(exit_action)
        # Exit action is always enabled
        # self.file_menu_actions.append(exit_action)


    def _create_view_menu(self, menubar: QMenuBar):
        """Creates the View menu."""
        view_menu = menubar.addMenu("&View")
        self.view_menu_actions = []

        # Theme Submenu
        theme_menu = view_menu.addMenu("Themes")
        self._populate_theme_menu(theme_menu) # Populates self.theme_actions

        # Panels Submenu (actions added when docks created)
        self.panels_menu = view_menu.addMenu("Panels")
        # Store reference to menu for later dock action additions

        view_menu.addSeparator()

        # Reset Zoom Action (Connection moved to toolbar creation where plot_widget exists)
        self.reset_zoom_action = self._create_action( "Reset Zoom", "zoom-original", "Reset plot zoom and pan", initial_enabled=False, shortcut="Ctrl+H")
        view_menu.addAction(self.reset_zoom_action)
        self.view_menu_actions.append(self.reset_zoom_action)

    def _populate_theme_menu(self, theme_menu: QMenu):
        """Populates the theme selection submenu."""
        theme_menu.clear()
        self.theme_actions.clear() # Clear previous actions if repopulating
        available_themes = self.theme_manager.get_available_themes()

        if not available_themes:
            no_themes_action = QAction("No themes found", self)
            no_themes_action.setEnabled(False)
            theme_menu.addAction(no_themes_action)
            return

        current_theme = self.theme_manager.current_theme_name
        for theme_name in available_themes:
            action = QAction(theme_name.replace('_', ' ').title(), self, checkable=True)
            action.setChecked(theme_name == current_theme)
            # Use lambda with default argument capture
            action.triggered.connect(lambda checked=False, name=theme_name: self.change_theme(name))
            self.theme_actions[theme_name] = action
            theme_menu.addAction(action)
        # Add theme actions to main list for potential future global enable/disable
        self.view_menu_actions.extend(self.theme_actions.values())

    def _create_tools_menu(self, menubar: QMenuBar):
        """Creates the Tools menu."""
        tools_menu = menubar.addMenu("&Tools")
        self.tools_menu_actions = []

        fetch_nist_action = self._create_action( "Fetch NIST Data (Script)...", "download", "Run script to download atomic data from NIST", self.run_nist_fetcher)
        tools_menu.addAction(fetch_nist_action)
        self.tools_menu_actions.append(fetch_nist_action)

        build_data_action = self._create_action( "Build Atomic Data Files (Script)...", "document-properties", "Run script to build U(T)/V_ion files (Requires User Implementation!)", self.run_atomic_data_builder)
        tools_menu.addAction(build_data_action)
        self.tools_menu_actions.append(build_data_action)
        # Add other tools as needed

    def _create_help_menu(self, menubar: QMenuBar):
        """Creates the Help menu."""
        help_menu = menubar.addMenu("&Help")
        # No need to add these to enable/disable list

        about_act = self._create_action( f"About {QCoreApplication.applicationName()}", "help-about", f"Show information about {QCoreApplication.applicationName()}", self.show_about_dialog)
        help_menu.addAction(about_act)

        online_docs_act = self._create_action( "Online Documentation", "help-contents", "Open online documentation (GitHub)", self._open_online_docs)
        help_menu.addAction(online_docs_act)

    def _create_tool_bar(self):
        """Creates the main application toolbar."""
        self.main_toolbar = QToolBar("Main Toolbar", self)
        self.main_toolbar.setIconSize(QSize(24, 24)) # Consistent icon size
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.main_toolbar)
        self.toolbar_actions = [] # Store actions for enable/disable

        # --- Add Common Actions ---
        if self.load_action:
            self.load_action.setToolTip("Load single spectrum (Ctrl+O)") # Tooltips are important for toolbars
            self.main_toolbar.addAction(self.load_action)
            self.toolbar_actions.append(self.load_action)
        # Add Multi-Load? Maybe too cluttered. Keep in File menu for now.
        # if self.load_multi_action: self.main_toolbar.addAction(self.load_multi_action)

        # Save Session Action
        if self.save_session_action:
             self.save_session_action.setToolTip("Save current session state (Ctrl+S)")
             self.main_toolbar.addAction(self.save_session_action)
             self.toolbar_actions.append(self.save_session_action)

        # Save Plot Action (different instance for toolbar)
        self.save_plot_toolbar_action = self._create_action( "Save Plot", "document-save", "Save the current plot as an image", lambda: self._save_action(SaveType.PLOT), initial_enabled=False)
        self.main_toolbar.addAction(self.save_plot_toolbar_action)
        self.toolbar_actions.append(self.save_plot_toolbar_action)

        self.main_toolbar.addSeparator()

        # --- Add Plot Navigation Actions ---
        if self.plot_widget and hasattr(self.plot_widget, 'toolbar') and self.plot_widget.toolbar:
            nav_toolbar = self.plot_widget.toolbar
            # Try to find standard actions robustly
            home_action = next((a for a in nav_toolbar.actions() if "home" in a.toolTip().lower()), None)
            pan_action = next((a for a in nav_toolbar.actions() if "pan" in a.toolTip().lower()), None)
            zoom_action = next((a for a in nav_toolbar.actions() if "zoom" in a.toolTip().lower()), None)

            # Use our custom reset zoom action if defined, otherwise fallback to matplotlib's
            if self.reset_zoom_action:
                 self.reset_zoom_action.setToolTip("Reset Zoom (Ctrl+H)")
                 self.main_toolbar.addAction(self.reset_zoom_action)
                 # Connect its trigger here where we know plot_widget exists
                 self.reset_zoom_action.triggered.connect(nav_toolbar.home)
                 self.toolbar_actions.append(self.reset_zoom_action)
            elif home_action:
                home_action.setToolTip("Reset Zoom (Ctrl+H)") # Standardize tooltip
                home_action.setIcon(get_icon("zoom-original")) # Use consistent icon
                self.main_toolbar.addAction(home_action)
                self.toolbar_actions.append(home_action)
            else: log.warning("Could not find Reset Zoom/Home action in plot toolbar.")

            if pan_action:
                pan_action.setToolTip("Pan/Move Plot (Hold middle mouse button)") # More specific tooltip
                pan_action.setIcon(get_icon("transform-move")) # Example theme icon
                self.main_toolbar.addAction(pan_action)
                self.toolbar_actions.append(pan_action)
            else: log.warning("Could not find Pan action in plot toolbar.")

            if zoom_action:
                zoom_action.setToolTip("Zoom Box (Hold right mouse button)") # More specific tooltip
                zoom_action.setIcon(get_icon("zoom-fit-best")) # Example theme icon
                self.main_toolbar.addAction(zoom_action)
                self.toolbar_actions.append(zoom_action)
            else: log.warning("Could not find Zoom action in plot toolbar.")
        else:
            log.warning("Plot widget or its toolbar not available when creating main toolbar. Navigation actions omitted.")
            # Add our reset zoom action anyway, but disabled
            if self.reset_zoom_action:
                 self.reset_zoom_action.setToolTip("Reset Zoom (Ctrl+H)")
                 self.reset_zoom_action.setEnabled(False)
                 self.main_toolbar.addAction(self.reset_zoom_action)
                 self.toolbar_actions.append(self.reset_zoom_action)

    def _create_dock_widgets(self):
        """Creates and arranges all the dockable panels."""
        self.docks = {}
        # Use Enum members for keys
        left_area = Qt.DockWidgetArea.LeftDockWidgetArea
        right_area = Qt.DockWidgetArea.RightDockWidgetArea
        bottom_area = Qt.DockWidgetArea.BottomDockWidgetArea

        # --- Instantiate Panel Widgets ---
        # Wrap instantiation in try/except to catch errors early
        try:
            self.processing_panel = ProcessingControlPanel(self.config, self)
            self.peak_detection_panel = PeakDetectionControlPanel(self.config, self)
            self.peak_fitting_panel = PeakFittingControlPanel(self.config, self)
            self.peak_list_view = PeakListView(self)
            self.nist_search_view = NistSearchView(self.config, self)
            self.boltzmann_view = BoltzmannPlotView(self.config, self)
            self.cf_libs_view = CfLibsView(self.config, self)
            self.ml_view = MLAnalysisView(self.config, self)
        except Exception as e:
            log.critical(f"Failed to instantiate one or more panel widgets: {e}", exc_info=True)
            QMessageBox.critical(self, "UI Initialization Error", f"Failed to create control panels:\n{e}")
            # Prevent further dock creation if panels are missing
            return

        # --- Helper Function to Add Docks ---
        # (Using Enum members for keys/names)
        def add_dock(
            name: DockName, title: str, widget: Optional[QWidget], area: Qt.DockWidgetArea,
            shortcut_num: int, tabify_with: Optional[QDockWidget] = None
        ) -> Optional[QDockWidget]:
            if widget is None: return None # Skip if widget failed instantiation

            dock = QDockWidget(title, self)
            dock.setObjectName(f"{name.value}Dock") # Use enum value for object name
            dock.setWidget(widget)
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                             QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                             QDockWidget.DockWidgetFeature.DockWidgetClosable)

            self.addDockWidget(area, dock)
            self.docks[name] = dock # Store using Enum member as key

            # Add toggle action to View -> Panels menu
            if self.panels_menu:
                 toggle_action = dock.toggleViewAction()
                 toggle_action.setText(title) # Use readable title
                 toggle_action.setShortcut(f"Ctrl+{shortcut_num}")
                 self.panels_menu.addAction(toggle_action)
                 self.view_menu_actions.append(toggle_action) # Track for enable/disable
            else:
                 log.error("Panels menu not initialized before adding dock toggle action.")

            # Tabify if requested
            if tabify_with:
                # Ensure tabify_with is a QDockWidget instance
                if isinstance(tabify_with, QDockWidget):
                     self.tabifyDockWidget(tabify_with, dock)
                else:
                     log.warning(f"Cannot tabify dock '{name.value}': 'tabify_with' is not a QDockWidget.")

            # Widget starts enabled=False by default in add_dock call
            # Let _update_action_panel_states control initial state
            # widget.setEnabled(False) # Removed initial disable here
            return dock

        # --- Create and Add Docks using Enums ---
        proc_dock = add_dock(DockName.PROCESSING, '1. Processing', self.processing_panel, left_area, 1)
        detect_dock = add_dock(DockName.DETECTION, '2. Detection', self.peak_detection_panel, left_area, 2, proc_dock)
        fit_dock = add_dock(DockName.FITTING, '3. Fitting', self.peak_fitting_panel, left_area, 3, detect_dock)

        list_dock = add_dock(DockName.PEAK_LIST, 'Peak List', self.peak_list_view, right_area, 4)
        nist_dock = add_dock(DockName.NIST_SEARCH, 'NIST Search', self.nist_search_view, right_area, 5, list_dock)

        boltzmann_dock = add_dock(DockName.BOLTZMANN, 'Boltzmann', self.boltzmann_view, bottom_area, 6)
        cflibs_dock = add_dock(DockName.CFLIBS, 'CF-LIBS', self.cf_libs_view, bottom_area, 7, boltzmann_dock)
        ml_dock = add_dock(DockName.ML_ANALYSIS, 'ML Analysis', self.ml_view, bottom_area, 8, cflibs_dock)

        # --- Initial Dock Visibility / Focus ---
        # Raise the first tab in each tabified group
        if proc_dock: proc_dock.raise_()
        if list_dock: list_dock.raise_()
        if boltzmann_dock: boltzmann_dock.raise_()


    def _connect_signals(self):
        """Connects signals from UI elements to slots in the main window."""
        log.debug("Connecting MainWindow signals...")

        # Processing Panel
        if self.processing_panel: self.processing_panel.process_triggered.connect(self.handle_process_request)
        # Peak Detection Panel
        if self.peak_detection_panel: self.peak_detection_panel.detect_peaks_triggered.connect(self.handle_peak_detection_request)
        # Peak Fitting Panel
        if self.peak_fitting_panel:
            self.peak_fitting_panel.fit_peaks_triggered.connect(self.handle_peak_fitting_request)
            self.peak_fitting_panel.refit_single_peak_requested.connect(self.handle_refit_single_peak)
            self.peak_fitting_panel.show_specific_fit.connect(self.handle_show_specific_fit)
        # Peak List View
        if self.peak_list_view: self.peak_list_view.peak_selected.connect(self.handle_peak_selection)
        # Plot Widget
        if self.plot_widget: self.plot_widget.peak_clicked.connect(self.handle_peak_selection) # Connect plot click directly to selection handler
        # NIST Search View
        if self.nist_search_view: self.nist_search_view.online_results_obtained.connect(self._handle_nist_search_results)
        # Boltzmann View
        if self.boltzmann_view:
            self.boltzmann_view.populate_lines_requested.connect(self.handle_boltzmann_populate_request)
            self.boltzmann_view.calculation_complete.connect(self._handle_boltzmann_result)
        # CF-LIBS View
        if self.cf_libs_view:
            self.cf_libs_view.calculate_ne_requested.connect(self.handle_ne_calculation_request)
            self.cf_libs_view.calculate_conc_requested.connect(self.handle_conc_calculation_request)
        # ML Analysis View
        if self.ml_view: self.ml_view.status_update.connect(self.update_status) # Connect to status bar

        # Connect theme actions (already done in _populate_theme_menu)


    # --- Utility / Helper Methods ---

    def _create_action(self, text: str, icon_name: Optional[str] = None,
                      status_tip: Optional[str] = None, triggered_slot: Optional[Callable] = None,
                      initial_enabled: bool = True, shortcut: Optional[Union[QKeySequence, QKeySequence.StandardKey, str]] = None,
                      tool_tip: Optional[str] = None, checkable: bool = False) -> QAction:
        """Helper to create a QAction with common settings."""
        # Use self as parent for actions tied to the main window lifecycle
        icon = get_icon(icon_name) if icon_name else QIcon()
        action = QAction(icon, text, self) # Set self as parent
        if shortcut: action.setShortcut(QKeySequence(shortcut))
        if status_tip: action.setStatusTip(status_tip)
        # Use status tip as tooltip if tooltip not provided
        final_tooltip = tool_tip if tool_tip else status_tip
        if final_tooltip: action.setToolTip(final_tooltip)
        if triggered_slot: action.triggered.connect(triggered_slot)
        action.setEnabled(initial_enabled)
        action.setCheckable(checkable)
        return action

    def _get_critical_widgets_for_busy_state(self) -> List[QWidget]:
        """Returns a list of widgets/actions to disable during busy operations."""
        widgets = [self.menubar, self.main_toolbar] # Disable menus and toolbar
        # Add dock widgets themselves (disabling the widget inside is often better)
        # for dock in self.docks.values(): widgets.append(dock)
        # Add panels explicitly if disabling the dock isn't sufficient
        panels = [
             self.processing_panel, self.peak_detection_panel, self.peak_fitting_panel,
             self.peak_list_view, self.nist_search_view, self.boltzmann_view,
             self.cf_libs_view, self.ml_view
        ]
        widgets.extend([p for p in panels if p is not None])
        # Add key actions if needed (toolbar actions often covered by disabling toolbar)
        # widgets.extend([a for a in self.file_menu_actions if a and a.isEnabled()]) # Example
        return widgets


    def set_busy(self, busy: bool, message: str = "Working...", disable_widgets: Optional[List[QWidget]] = None):
        """Sets the application's busy state, updates cursor/status, and disables key widgets."""
        if busy == self._is_busy: return # Avoid redundant calls

        self._is_busy = busy
        widgets_to_manage = disable_widgets if disable_widgets is not None else self._get_critical_widgets_for_busy_state()

        if busy:
            log.debug(f"Setting busy state ON: {message}")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.update_status(message, timeout=0) # Show indefinitely
            # Disable critical UI elements
            for widget in widgets_to_manage:
                 if widget: widget.setEnabled(False)
        else:
            log.debug(f"Setting busy state OFF.")
            QApplication.restoreOverrideCursor()
            # Re-enable critical UI elements (let _update_action_panel_states handle final state)
            for widget in widgets_to_manage:
                  if widget: widget.setEnabled(True)
            # Update status to Ready only if no other timed message is active
            # This requires checking if statusbar has a current message. Simpler: just set Ready.
            self.update_status("Ready.") # Reset status to Ready

        # Process events minimally AFTER changing cursor/state
        QApplication.processEvents(flags=QCoreApplication.ProcessEventsFlag.ExcludeUserInputEvents) # Avoid processing user input during state change


    def update_status(self, message: str, timeout: int = 5000):
        """Updates the status bar message."""
        if self.statusBar() is None: # Check if status bar exists
             log.warning("Status bar not initialized. Ignoring status update.")
             return

        if timeout <= 0:
            # Show permanent message on the right label
            if self.status_label: self.status_label.setText(message)
            self.statusBar().clearMessage() # Clear temporary area
            log_msg = f"Status: {message} (Permanent)"
        else:
            # Show temporary message in main status bar area
            self.statusBar().showMessage(message, timeout)
            # Optionally clear permanent label? Or leave it? Leaving it is less disruptive.
            # if self.status_label: self.status_label.setText("")
            log_msg = f"Status: {message} (Timeout: {timeout}ms)"

        # Use lower log level for status updates unless it's an error
        log_level = logging.INFO if "error" in message.lower() or "fail" in message.lower() else logging.DEBUG
        log.log(log_level, log_msg)


    def show_about_dialog(self):
        """Displays the About dialog box with version info."""
        try:
            from PyQt6 import __version__ as PYQT_VERSION_STR
        except ImportError: PYQT_VERSION_STR = "N/A"
        try: import scipy; SCIPY_VERSION = scipy.__version__
        except ImportError: SCIPY_VERSION = "N/A"
        try: import matplotlib; MATPLOTLIB_VERSION = matplotlib.__version__
        except ImportError: MATPLOTLIB_VERSION = "N/A"
        try: import sklearn; SKLEARN_VERSION = sklearn.__version__
        except ImportError: SKLEARN_VERSION = "N/A"
        try: import PyWavelets; PYWT_VERSION = PyWavelets.__version__
        except ImportError: PYWT_VERSION = "N/A"
        try: import astroquery; AQ_VERSION = astroquery.__version__
        except ImportError: AQ_VERSION = "N/A"


        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        app_name = QCoreApplication.applicationName()
        app_version = QCoreApplication.applicationVersion()
        org_name = QCoreApplication.organizationName()
        project_root_str = str(get_project_root()) if get_project_root() else "N/A"

        about_text = f"""
        <h2>{app_name} v{app_version}</h2>
        <p>LIBS Data Analysis Suite.</p>
        <p>Developed by: {org_name}</p>
        <hr>
        <p><b>Environment:</b></p>
        <ul>
            <li>Python: {python_version}</li>
            <li>PyQt: {PYQT_VERSION_STR}</li>
        </ul>
        <p><b>Core Libraries:</b></p>
        <ul>
            <li>NumPy: {np.__version__ if np else 'N/A'}</li>
            <li>SciPy: {SCIPY_VERSION}</li>
            <li>Pandas: {pd.__version__ if pd else 'N/A'}</li>
            <li>Matplotlib: {MATPLOTLIB_VERSION}</li>
            <li>Astroquery: {AQ_VERSION}</li>
            <li>Scikit-learn: {SKLEARN_VERSION}</li>
            <li>PyYAML: {yaml.__version__ if yaml else 'N/A'}</li>
            <li>PyWavelets: {PYWT_VERSION}</li>
        </ul>
        <p>Project Root: {project_root_str}</p>
        <p><i>Copyright © 2024 {org_name}. All rights reserved.</i></p>
        """
        QMessageBox.about(self, f"About {app_name}", about_text)


    def _open_online_docs(self):
        """Opens the online documentation URL (e.g., GitHub README) in the default browser."""
        # Replace with your actual documentation URL from config or constant
        url_string = self.config.get('application',{}).get('documentation_url', "https://github.com/CosmicForge/libs-cosmic-forge#readme")
        url = QUrl(url_string)
        log.info(f"Attempting to open documentation URL: {url.toString()}")
        if not QDesktopServices.openUrl(url):
            log.error(f"Could not open URL: {url.toString()}")
            QMessageBox.warning(self, "Cannot Open Link",
                                f"Could not open the documentation link:\n{url.toString()}\n\nPlease open it manually in your browser.")


    def change_theme(self, theme_name: str):
        """Applies the selected theme and updates UI."""
        # Prevent changing theme while busy? Maybe not necessary.
        log.info(f"Changing theme to: {theme_name}")
        if theme_name not in self.theme_manager.get_available_themes():
            log.warning(f"Attempted to switch to non-existent theme: {theme_name}")
            return

        success = self.theme_manager.apply_theme(theme_name)
        if success:
            self._update_theme_menu_state() # Update menu checks
            self._apply_theme_to_plots() # Update plot colors
            self._save_theme_setting(theme_name) # Persist choice via QSettings
        else:
            log.error(f"Theme manager failed to apply theme: {theme_name}")
            QMessageBox.warning(self, "Theme Error", f"Could not apply the theme '{theme_name}'. Check logs.")


    def _apply_theme_to_plots(self):
        """Applies the current theme's colors to relevant plot widgets."""
        log.debug("Applying theme colors to plots.")
        # Gather all potential plot widgets
        plots_to_update = []
        if self.plot_widget and hasattr(self.plot_widget, 'apply_theme_colors'): plots_to_update.append(self.plot_widget)
        if self.boltzmann_view and hasattr(self.boltzmann_view, 'boltzmann_plot_widget') and getattr(self.boltzmann_view, 'boltzmann_plot_widget', None):
             plot = getattr(self.boltzmann_view, 'boltzmann_plot_widget')
             if hasattr(plot, 'apply_theme_colors'): plots_to_update.append(plot)
        if self.ml_view and hasattr(self.ml_view, 'results_plot_widget') and getattr(self.ml_view, 'results_plot_widget', None):
             plot = getattr(self.ml_view, 'results_plot_widget')
             if hasattr(plot, 'apply_theme_colors'): plots_to_update.append(plot)

        theme_applied_count = 0
        for plot in plots_to_update:
            try:
                plot.apply_theme_colors(self.config) # Pass config for color lookup
                theme_applied_count += 1
            except Exception as e:
                log.error(f"Error applying theme to plot {type(plot).__name__}: {e}", exc_info=True)
        log.debug(f"Applied theme colors to {theme_applied_count} plot widgets.")


    def _update_theme_menu_state(self):
        """Updates the check state of the theme menu actions."""
        current_theme = self.theme_manager.current_theme_name
        if not self.theme_actions: return # Check if dict is populated
        for name, action in self.theme_actions.items():
            action.setChecked(name == current_theme)


    # --- Settings Persistence ---

    def _load_persistent_settings(self):
        """Loads window geometry, state, paths, and theme from QSettings."""
        # Check config first if persistence should be skipped
        if not self.config.get('application', {}).get('remember_window_state', True):
            log.info("Window state persistence is disabled in config. Skipping QSettings load.")
            # Apply default theme from config if persistence is off
            default_theme_name = self.config.get('appearance', {}).get('default_theme', DEFAULT_THEME)
            log.info(f"Applying configured default theme: {default_theme_name}")
            self.change_theme(default_theme_name) # Apply theme and update plots/menus
            return

        log.info("Loading persistent window settings using QSettings...")
        try:
            # Geometry
            geom_value = self.settings.value("MainWindow/geometry")
            if geom_value is not None:
                 if isinstance(geom_value, QByteArray) and not geom_value.isNull():
                     if self.restoreGeometry(geom_value): log.debug("Restored window geometry.")
                     else: log.warning("Failed to restore window geometry from settings.")
                 elif isinstance(geom_value, bytes): # Handle raw bytes if saved differently
                      if self.restoreGeometry(QByteArray(geom_value)): log.debug("Restored window geometry from bytes.")
                      else: log.warning("Failed to restore window geometry from settings (bytes).")
                 else: log.warning(f"Invalid geometry data type in settings: {type(geom_value)}")
            else: log.debug("No geometry found in settings.")

            # Window State (Docks, Toolbars)
            state_value = self.settings.value("MainWindow/windowState")
            if state_value is not None:
                 if isinstance(state_value, QByteArray) and not state_value.isNull():
                     if self.restoreState(state_value): log.debug("Restored window state (docks/toolbars).")
                     else: log.warning("Failed to restore window state from settings.")
                 elif isinstance(state_value, bytes):
                      if self.restoreState(QByteArray(state_value)): log.debug("Restored window state from bytes.")
                      else: log.warning("Failed to restore window state from settings (bytes).")
                 else: log.warning(f"Invalid window state data type in settings: {type(state_value)}")
            else: log.debug("No window state found in settings.")

            # Paths (use fallback from _init_ui_state if setting missing)
            self._last_save_dir = self.settings.value("Paths/lastSaveDir", defaultValue=self._last_save_dir)
            self._last_load_dir = self.settings.value("Paths/lastLoadDir", defaultValue=self._last_load_dir)
            log.debug(f"Restored last dirs: Load='{self._last_load_dir}', Save='{self._last_save_dir}'")

            # Theme
            last_theme_name = self._load_last_theme() # Handles fallback to config default
            log.info(f"Applying restored/default theme: {last_theme_name}")
            self.change_theme(last_theme_name) # Applies theme and updates plots/menus

        except Exception as e:
            log.error(f"Failed to load persistent window settings: {e}", exc_info=True)
            # Avoid resetting everything, just log the error

    def _save_persistent_settings(self):
        """Saves window geometry, state, paths, and theme to QSettings."""
        if not self.config.get('application', {}).get('remember_window_state', True):
            log.info("Window state persistence is disabled. Skipping QSettings save.")
            return

        log.info("Saving persistent window settings using QSettings...")
        try:
            self.settings.setValue("MainWindow/geometry", self.saveGeometry())
            self.settings.setValue("MainWindow/windowState", self.saveState())
            self.settings.setValue("Paths/lastSaveDir", self._last_save_dir)
            self.settings.setValue("Paths/lastLoadDir", self._last_load_dir)
            if hasattr(self, 'theme_manager'):
                 self.settings.setValue("Appearance/lastTheme", self.theme_manager.current_theme_name)

            # Explicitly sync to write changes
            self.settings.sync()
            status = self.settings.status()
            if status != QSettings.Status.NoError:
                log.error(f"QSettings sync error while saving window settings: Status {status}")
            else:
                 log.debug("Window settings synced successfully.")
        except Exception as e:
            log.error(f"Failed to save persistent window settings: {e}", exc_info=True)

    def _save_theme_setting(self, theme_name: str):
        """Saves only the theme setting immediately."""
        if not self.config.get('application', {}).get('remember_window_state', True): return
        try:
            self.settings.setValue("Appearance/lastTheme", theme_name)
            self.settings.sync()
            status = self.settings.status()
            if status != QSettings.Status.NoError: log.error(f"QSettings sync error saving theme: {status}")
            else: log.debug(f"Persisted theme setting: {theme_name}")
        except Exception as e: log.error(f"Failed to save theme setting: {e}", exc_info=True)

    def _load_last_theme(self) -> str:
        """Loads the last used theme from settings, falling back to config default."""
        config_default_theme = self.config.get('appearance', {}).get('default_theme', DEFAULT_THEME)
        if not self.config.get('application', {}).get('remember_window_state', True):
             return config_default_theme
        try:
            last_theme = self.settings.value("Appearance/lastTheme", defaultValue=config_default_theme)
            # Validate the loaded theme exists
            if last_theme and last_theme in self.theme_manager.get_available_themes():
                return last_theme
            else:
                log.warning(f"Saved theme '{last_theme}' not found or invalid. Using default '{config_default_theme}'.")
                return config_default_theme
        except Exception as e:
            log.error(f"Failed to load last theme from settings: {e}. Using default.", exc_info=True)
            return config_default_theme

    # --- Application State Management ---

    def _mark_dirty(self, is_dirty: bool = True):
        """Sets the application's 'dirty' state (unsaved changes)."""
        if is_dirty != self._is_dirty:
             self._is_dirty = is_dirty
             log.debug(f"Application dirty state set to: {self._is_dirty}")
             # Update window title to reflect state (add/remove asterisk)
             title = self.windowTitle()
             if self._is_dirty and not title.endswith("*"):
                  self.setWindowTitle(title + "*")
             elif not self._is_dirty and title.endswith("*"):
                  self.setWindowTitle(title[:-1])

    def _clear_core_analysis_data(self):
         """Resets core data attributes related to analysis results."""
         log.debug("Clearing core analysis data (peaks, matches, plasma, dataframes).")
         self.detected_peaks = []
         self.nist_matches = []
         self.plasma_temp_k = None
         self.electron_density_cm3 = None
         self.boltzmann_plot_data = None
         self.cf_libs_concentrations = None

    def _clear_panel_displays(self):
        """Calls clear_all() or reset methods on relevant panels."""
        log.debug("Clearing panel displays.")
        # List panels that need clearing
        panels_to_clear = [
            self.peak_list_view, self.peak_fitting_panel, self.nist_search_view,
            self.boltzmann_view, self.cf_libs_view, self.ml_view
        ]
        for panel in panels_to_clear:
             if panel and hasattr(panel, 'clear_all') and callable(panel.clear_all):
                 try: panel.clear_all()
                 except Exception as e: log.error(f"Error calling clear_all for {type(panel).__name__}: {e}")
             elif panel and hasattr(panel, 'clear_results') and callable(panel.clear_results):
                  try: panel.clear_results() # Handle panels with different clear method names
                  except Exception as e: log.error(f"Error calling clear_results for {type(panel).__name__}: {e}")
             # Add other clear methods if needed
             elif panel:
                  log.debug(f"Panel {type(panel).__name__} does not have a standard clear method.")


    def _reset_state_for_new_spectrum(self, spectrum: Optional[Spectrum]):
        """
        Resets application state, optionally loading a new single spectrum.
        Clears multi-spectra list if loading a single spectrum or clearing all.
        """
        is_clearing_all = spectrum is None and not self.multi_spectra
        is_loading_single = spectrum is not None
        is_multi_mode_active = bool(self.multi_spectra) and not is_loading_single

        log.info(f"Resetting application state. Loading single: {is_loading_single}, "
                 f"Multi-mode active: {is_multi_mode_active}, Clearing all: {is_clearing_all}.")

        # Clear core data
        self.current_spectrum = spectrum # Set new spectrum (or None)
        self._clear_core_analysis_data()

        # Clear multi-spectra list if switching to single or clearing all
        if is_loading_single or is_clearing_all:
            if self.multi_spectra: log.debug("Clearing multi-spectra list.")
            self.multi_spectra = []
            # Explicitly clear ML view if it was in multi-mode
            if self.ml_view:
                if hasattr(self.ml_view, 'set_spectra_list'): self.ml_view.set_spectra_list([])
                if hasattr(self.ml_view, 'clear_all'): self.ml_view.clear_all()

        # --- Update Title and Status ---
        status_msg = "Ready."
        window_title = f"{QCoreApplication.applicationName()} v{QCoreApplication.applicationVersion()}"
        if self.current_spectrum:
            try:
                fname = os.path.basename(self.current_spectrum.filename) if self.current_spectrum.filename else "Untitled Spectrum"
                points = len(self.current_spectrum)
                status_msg = f"Loaded: {fname} ({points} points)"
                window_title = f"{QCoreApplication.applicationName()} - {fname}"
            except AttributeError:
                 log.error("Invalid Spectrum object encountered during state reset.")
                 status_msg = "Error: Invalid spectrum data."
                 self.current_spectrum = None # Ensure it's None if invalid
        elif self.multi_spectra:
            status_msg = f"Ready for ML analysis ({len(self.multi_spectra)} spectra loaded)."
            window_title = f"{QCoreApplication.applicationName()} - Multiple Spectra ({len(self.multi_spectra)})"
        else:
            status_msg = "State cleared. No spectrum loaded."

        self.update_status(status_msg)
        self.setWindowTitle(window_title)

        # --- Clear Panel Displays ---
        self._clear_panel_displays()

        # --- Clear Plot ---
        if self.plot_widget:
            self.plot_widget.clear_plot() # Clear plot fully
            if self.current_spectrum:
                # Plot the new single spectrum (raw only initially?)
                self.plot_widget.plot_spectrum(self.current_spectrum, show_raw=True, show_processed=False, show_baseline=False)
            elif self.multi_spectra:
                 # Optionally plot overview or first spectrum for multi-mode?
                 # self.plot_widget.plot_spectrum(self.multi_spectra[0], ...) # Example
                 pass # Or leave plot empty for multi-mode initial state
            # Apply theme after potential replot
            self._apply_theme_to_plots()

        # Mark state as clean after loading/clearing
        self._mark_dirty(False)
        # Update UI enablement based on the new state
        # Use the master update function which includes enablement checks
        self._update_ui_from_current_state()


    def _clear_downstream_analysis_data(self, clear_peaks: bool = True, clear_fits: bool = True,
                                        clear_nist: bool = True, clear_plasma: bool = True):
        """
        Clears specific analysis results and dependent UI elements.
        Hierarchical: Clearing peaks implies clearing fits, NIST, plasma.
                      Clearing fits implies clearing NIST, plasma.
                      Clearing NIST implies clearing plasma.
        """
        log.debug(f"Clearing downstream data: Peaks={clear_peaks}, Fits={clear_fits}, NIST={clear_nist}, Plasma={clear_plasma}")
        made_change = False

        # Determine effective clearing based on hierarchy
        if clear_peaks: clear_fits = clear_nist = clear_plasma = True
        elif clear_fits: clear_nist = clear_plasma = True
        elif clear_nist: clear_plasma = True

        # Clear Peaks
        if clear_peaks and self.detected_peaks:
            self.detected_peaks = []
            if self.peak_list_view: self.peak_list_view.update_peak_list([])
            if self.plot_widget: self.plot_widget.plot_peaks([])
            log.debug("Cleared detected peaks.")
            made_change = True

        # Clear Fits (within existing peaks if not clearing peaks)
        if clear_fits:
            fit_cleared = False
            for peak in self.detected_peaks:
                 if peak.best_fit or peak.alternative_fits:
                     peak.best_fit = None
                     peak.alternative_fits = {}
                     fit_cleared = True
            if fit_cleared:
                 log.debug("Cleared fit results from existing peaks.")
                 # Update list view to remove fit info
                 if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
                 # Clear fit lines from plot
                 if self.plot_widget: self.plot_widget.plot_fit_lines([])
                 # Clear fitting panel display
                 if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(None, None)
                 made_change = True

        # Clear NIST Correlations (within existing peaks if not clearing peaks/fits)
        if clear_nist:
            nist_cleared = False
            for peak in self.detected_peaks:
                 if peak.potential_matches:
                      peak.potential_matches = []
                      nist_cleared = True
            if nist_cleared or self.nist_matches: # Also clear main list
                 self.nist_matches = []
                 if self.plot_widget: self.plot_widget.clear_nist_matches()
                 if self.nist_search_view:
                      self.nist_search_view.set_peaks_reference(self.detected_peaks) # Update reference
                      self.nist_search_view.clear_results()
                 # Update peak list view for correlation icons
                 if self.peak_list_view: self.peak_list_view.update_peak_list(self.detected_peaks)
                 log.debug("Cleared NIST matches and correlations.")
                 made_change = True

        # Clear Plasma Parameters
        if clear_plasma:
            plasma_cleared = False
            if self.plasma_temp_k is not None: self.plasma_temp_k = None; plasma_cleared = True
            if self.electron_density_cm3 is not None: self.electron_density_cm3 = None; plasma_cleared = True
            if self.boltzmann_plot_data is not None: self.boltzmann_plot_data = None; plasma_cleared = True
            if self.cf_libs_concentrations is not None: self.cf_libs_concentrations = None; plasma_cleared = True
            if plasma_cleared:
                 # Update display panels
                 if self.boltzmann_view: self.boltzmann_view.clear_all()
                 if self.cf_libs_view: self.cf_libs_view.clear_all()
                 log.debug("Cleared plasma parameters (T, Ne) and derived data (Boltzmann, CF-LIBS).")
                 made_change = True

        # Update UI states if any data was actually cleared
        if made_change:
            self._update_ui_from_current_state() # Use master update function


    def _handle_load_error(self, filepath: str, error: Any):
        """Handles errors during file loading and resets state."""
        error_str = str(error)
        log.error(f"Failed to load spectrum from '{os.path.basename(filepath)}': {error_str}", exc_info=True)
        QMessageBox.critical(
            self, "Spectrum Load Error",
            f"Error loading file:\n{os.path.basename(filepath)}\n\n"
            f"Details:\n{error_str}\n\n"
            f"Check file format, delimiter ({repr(self.default_delimiter)}), " # Show repr
            f"and comment character ({repr(self.default_comment_char)})."
        )
        # Reset state fully after load error
        self._reset_state_for_new_spectrum(None)


    def _update_action_panel_states(self):
        """Updates the enabled/disabled state of all actions and dock panels."""
        # Determine current state flags
        is_single_loaded = self.current_spectrum is not None
        is_multi_loaded = bool(self.multi_spectra)
        is_processed = is_single_loaded and self.current_spectrum.processed_intensity is not None
        has_peaks = bool(self.detected_peaks)
        has_fits = has_peaks and any(p.best_fit and p.best_fit.success for p in self.detected_peaks)
        has_nist_matches = bool(self.nist_matches)
        # NIST correlation requires fits AND having run correlation
        has_nist_correlation = has_fits and any(p.potential_matches for p in self.detected_peaks)
        has_temp = self.plasma_temp_k is not None and np.isfinite(self.plasma_temp_k)
        has_ne = self.electron_density_cm3 is not None and np.isfinite(self.electron_density_cm3)
        has_boltzmann_data = self.boltzmann_plot_data is not None and not self.boltzmann_plot_data.empty
        has_conc_data = self.cf_libs_concentrations is not None and not self.cf_libs_concentrations.empty
        can_plot = self.plot_widget is not None # Check if plot widget exists

        # --- Panel Enablement ---
        panel_states = {
            DockName.PROCESSING: is_single_loaded and not is_multi_loaded,
            DockName.DETECTION: is_single_loaded and is_processed and not is_multi_loaded,
            DockName.PEAK_LIST: is_single_loaded and not is_multi_loaded, # Show list even if empty
            DockName.FITTING: is_single_loaded and has_peaks and not is_multi_loaded,
            DockName.NIST_SEARCH: is_single_loaded and has_fits and not is_multi_loaded and ASTROQUERY_AVAILABLE, # Needs fits, check astroquery
            DockName.BOLTZMANN: is_single_loaded and has_nist_correlation and not is_multi_loaded, # Needs correlated NIST
            DockName.CFLIBS: is_single_loaded and has_nist_correlation and has_temp and not is_multi_loaded, # Needs T, maybe Ne later
            DockName.ML_ANALYSIS: is_multi_loaded and not is_single_loaded, # Only in multi mode
        }
        log.debug(f"Updating panel states: { {k.value: v for k, v in panel_states.items()} }")
        for name, dock in self.docks.items():
            widget = dock.widget()
            if widget:
                is_enabled = panel_states.get(name, False)
                # Check if widget has its own setEnabled override
                if hasattr(widget, 'setEnabledOverride'):
                     widget.setEnabledOverride(is_enabled) # Custom logic if needed
                else:
                     widget.setEnabled(is_enabled) # Default Qt logic

        # --- Action Enablement ---
        action_states = {
            # File Menu
            self.load_action: True,
            self.load_multi_action: True,
            self.load_session_action: True,
            self.save_session_action: is_single_loaded or is_multi_loaded,
            self.save_processed_action: is_processed,
            self.save_peaks_action: has_peaks,
            self.save_nist_action: has_nist_matches,
            self.save_boltzmann_action: has_boltzmann_data,
            self.save_conc_action: has_conc_data,
            self.save_plot_action: can_plot and (is_single_loaded or is_multi_loaded), # Enable if plot exists and data loaded
            # View Menu
            self.reset_zoom_action: can_plot,
            # Tools Menu (always enabled for now)
            # Toolbar Actions (mirror menu actions)
            self.save_plot_toolbar_action: can_plot and (is_single_loaded or is_multi_loaded),
            # Add other toolbar actions if needed
        }
        # Apply states to actions
        for action, enabled in action_states.items():
            if action: action.setEnabled(enabled)
        # Apply states to dock toggle actions in View menu
        for action in self.panels_menu.actions() if self.panels_menu else []: action.setEnabled(True) # Keep toggle always enabled
        # Apply states to theme actions (always enabled)
        for action in self.theme_actions.values(): action.setEnabled(True)

        # Enable/disable plot interaction based on peaks existing (for peak clicking)
        if self.plot_widget and hasattr(self.plot_widget, 'set_interaction_enabled'):
            self.plot_widget.set_interaction_enabled(has_peaks)

    def _update_ui_from_current_state(self):
        """MASTER UI UPDATE FUNCTION. Call this after any significant state change."""
        log.debug("Master UI Update: Updating plots, tables, and action/panel states.")
        # 1. Update Plot (handles multiple scenarios internally)
        if self.plot_widget:
             self.plot_widget.clear_plot(redraw=False) # Clear without intermediate draw
             if self.current_spectrum:
                  # Show all layers relevant to the current state
                  self.plot_widget.plot_spectrum(self.current_spectrum, show_raw=True,
                                                 show_processed=self.current_spectrum.processed_intensity is not None,
                                                 show_baseline=self.current_spectrum.baseline is not None)
                  self.plot_widget.plot_peaks(self.detected_peaks)
                  self.plot_widget.plot_fit_lines(self.detected_peaks, highlight_fit=self.plot_widget._highlighted_fit_result) # Keep highlight if valid
                  self.plot_widget.plot_nist_matches(self.nist_matches, correlate=False) # Plot correlated matches
             elif self.multi_spectra:
                   # Optional: Update ML plot if applicable, or show placeholder
                   if self.ml_view and self.ml_view._analysis_results is not None:
                        # Replot ML results maybe? Or handled by ML view itself.
                        pass
                   else:
                        self.plot_widget.ax.set_title("Multiple Spectra Loaded (ML Analysis)")
             else:
                  self.plot_widget.ax.set_title("Spectrum Plot") # Default title

             self._apply_theme_to_plots() # Ensure theme is applied
             self.plot_widget._redraw_canvas() # Redraw plot once at the end

        # 2. Update Peak List Table
        if self.peak_list_view:
             self.peak_list_view.update_peak_list(self.detected_peaks)
             # TODO: Restore selection if appropriate?

        # 3. Update NIST Search Table
        if self.nist_search_view and hasattr(self.nist_search_view, 'display_results'):
             if self.nist_matches:
                  # Regenerate DF for display if needed (safer than storing DF state)
                  try:
                       if hasattr(NISTMatch, 'to_dataframe_row'):
                            # Pass dummy peak info, as correlation is assumed done
                            dummy_peak_wl, dummy_peak_int = np.nan, np.nan
                            nist_df = pd.DataFrame([m.to_dataframe_row(dummy_peak_wl, dummy_peak_int) for m in self.nist_matches])
                            self.nist_search_view.display_results(nist_df)
                       else: log.warning("Cannot update NIST table: NISTMatch missing method.")
                  except Exception as e: log.error(f"Error creating NIST DF for display: {e}")
             else:
                  self.nist_search_view.display_results(pd.DataFrame()) # Display empty table

        # 4. Update Boltzmann View (e.g., title or result label if needed)
        if self.boltzmann_view:
            # Only update result label, population happens via user action
            temp_str = f"{self.plasma_temp_k:.0f} K" if self.plasma_temp_k is not None else "N/A"
            # Check R2 if available from boltzmann_plot_data? Complex. Keep simple.
            self.boltzmann_view.result_label.setText(f"Result: Tₑ = {temp_str}")

        # 5. Update CF-LIBS View (Temp, Ne, Concentrations)
        if self.cf_libs_view:
            self.cf_libs_view.update_temperature(self.plasma_temp_k)
            self.cf_libs_view.update_electron_density(self.electron_density_cm3)
            self.cf_libs_view.display_concentrations(self.cf_libs_concentrations)

        # 6. Update ML View (if needed)
        if self.ml_view:
             if self.multi_spectra and not self.ml_view._spectra_list: # Update if list is empty but multi spectra exist
                  self.ml_view.set_spectra_list(self.multi_spectra)
             elif not self.multi_spectra and self.ml_view._spectra_list: # Clear if no multi spectra
                  self.ml_view.set_spectra_list([])

        # 7. Update Action/Panel Enablement States
        self._update_action_panel_states()


    # --- Session Load/Save Trigger Slots ---

    @pyqtSlot()
    def _on_save_session_triggered(self) -> bool:
        """Handles the Save Session action, including file dialog."""
        if self._is_busy:
            log.warning("Save Session action ignored while busy.")
            self.update_status("Cannot save session: Busy.", 3000)
            return False
        if not self.current_spectrum and not self.multi_spectra:
            QMessageBox.information(self, "Save Session", "No spectrum or analysis data loaded to save.")
            return False

        log.info("Triggered Save Session action.")
        file_filter = f"{QCoreApplication.applicationName()} Session (*{SessionManager.SESSION_FILE_EXTENSION});;All Files (*)"

        # Suggest filename based on current spectrum or default
        suggested_name = f"libs_forge_session{SessionManager.SESSION_FILE_EXTENSION}"
        if self.current_spectrum and self.current_spectrum.filename:
            try:
                base_name = Path(self.current_spectrum.filename).stem
                suggested_name = f"{base_name}_session{SessionManager.SESSION_FILE_EXTENSION}"
            except Exception: pass # Ignore errors getting stem
        default_path = os.path.join(self._last_save_dir, suggested_name)

        # Use QFileDialog to get the save path
        filepath, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Session As...", default_path, file_filter
        )

        if not filepath:
            log.info("Save Session action cancelled by user.")
            self.update_status("Save session cancelled.", 3000)
            return False

        # Ensure correct extension is added if user didn't type it
        # Check based on the selected filter, not just the default extension
        expected_ext = SessionManager.SESSION_FILE_EXTENSION
        if f"(*{expected_ext})" in selected_filter and not filepath.lower().endswith(expected_ext):
            filepath += expected_ext
            log.debug(f"Appended session file extension: {filepath}")

        self._last_save_dir = os.path.dirname(filepath) # Remember directory
        # Disable critical UI during save
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, f"Saving session to {os.path.basename(filepath)}...", disable_widgets=widgets_to_disable)
        save_successful = False
        try:
            # Delegate saving logic to SessionManager
            save_successful = self.session_manager.save_session(filepath)
            if save_successful:
                self._mark_dirty(False) # Mark state as saved
                # Status update handled within save_session now
            else:
                # Handle case where save_session returns False without exception
                QMessageBox.warning(self, "Save Warning", "Session saving reported failure. Check logs for details.")
                # Status update handled within save_session
            return save_successful
        except Exception as e:
            log.error(f"Unexpected error during save session process: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Session Error", f"Could not save the session:\n{e}")
            self.update_status("Session save failed.", 5000)
            return False
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)


    @pyqtSlot()
    def _on_load_session_triggered(self):
        """Handles the Load Session action, including file dialog and state application."""
        # --- Prompt to save if dirty ---
        if not self._check_save_before_proceeding("load a new session"):
             return # User cancelled

        if self._is_busy:
            log.warning("Load Session action ignored while busy.")
            return

        log.info("Triggered Load Session action.")
        file_filter = f"{QCoreApplication.applicationName()} Session (*{SessionManager.SESSION_FILE_EXTENSION});;All Files (*)"
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Session", self._last_load_dir, file_filter
        )

        if not filepath:
            log.info("Load Session action cancelled by user.")
            self.update_status("Load session cancelled.", 3000)
            return

        self._last_load_dir = os.path.dirname(filepath)
        # Disable critical UI during load
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, f"Loading session from {os.path.basename(filepath)}...", disable_widgets=widgets_to_disable)

        try:
            # 1. Load data dictionary from file using SessionManager
            session_state = self.session_manager.load_session_data(filepath)

            # 2. Apply the loaded state (handles internal errors and state reset)
            if session_state:
                self._apply_loaded_session_state(session_state)
                # apply_loaded_state handles status updates internally based on success/failure
                log.info(f"Session loaded and applied from {filepath}.")
                self._mark_dirty(False) # Loaded session is considered clean initially
            else:
                # load_session_data should raise error, but handle None just in case
                raise ValueError("Loaded session state was empty or invalid.")

        except (FileNotFoundError, ValueError, IOError, KeyError, Exception) as e:
            log.error(f"Error loading or applying session file {filepath}: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Session Error", f"Error loading session file:\n{e}\n\nApplication state will be reset.")
            self.update_status("Session load failed.", 5000)
            self._reset_state_for_new_spectrum(None) # Reset fully on error
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable) # Re-enable UI


    def _apply_loaded_session_state(self, state: Dict[str, Any]):
        """
        Applies the loaded session state dictionary to the application.
        Handles internal errors and ensures UI is updated.
        """
        log.info("Applying loaded session state...")
        # Reset state first, but keep window geometry/state restoration separate
        # Clear core data and panels
        self._clear_core_analysis_data()
        self._clear_panel_displays()
        self.current_spectrum = None
        self.multi_spectra = []

        # Temporary flag to track if we should reset fully on error during apply
        apply_success = True
        try:
            # 1. Restore Window Geometry and State FIRST
            self._restore_window_settings(state)

            # 2. Restore Paths and Theme SECOND
            self._restore_paths_theme(state) # Applies theme and updates UI

            # 3. Reload Spectrum/Spectra Data from paths
            reloaded_single, reloaded_multi = self._reload_spectra_from_session(state)

            # CRITICAL: Check if required data was loaded before proceeding
            if state.get('current_spectrum_path') and not reloaded_single:
                 log.error("Session specified a single spectrum, but failed to reload it. Aborting analysis state restore.")
                 # Keep window/theme/paths, but clear analysis data
                 apply_success = False # Mark as failed
                 raise ValueError(f"Failed to reload spectrum file: {state.get('current_spectrum_path')}")
            elif state.get('multi_spectra_paths') and not reloaded_multi:
                 log.warning("Session specified multiple spectra, but failed to reload some/all. Proceeding with loaded data.")
                 # Allow partial restore for multi-spectra? Or fail? Let's proceed for now.

            # Update internal state *after* successful reload
            self.current_spectrum = reloaded_single
            self.multi_spectra = reloaded_multi

            # 4. Restore Analysis Results (Peaks, NIST, Plasma) - ONLY if single spectrum loaded
            if self.current_spectrum:
                self._restore_analysis_results(state) # Handles internal errors
            elif self.multi_spectra:
                 # Restore ML specific state if saved/needed
                 pass

            # 5. Restore Panel Settings (AFTER data/analysis restore)
            self._restore_panel_settings(state)

            # 6. Update UI to reflect the fully restored state
            # This will plot data, populate tables etc.
            self._update_ui_from_current_state() # Use master update
            self.update_status("Session loaded successfully.", 5000)

        except Exception as e:
            apply_success = False # Mark apply as failed
            log.error(f"Critical error applying loaded session state: {e}", exc_info=True)
            QMessageBox.critical(self, "Session Load Error", f"Failed to apply the loaded session state:\n{e}\n\nThe application state may be inconsistent and will be reset.")
            # Reset fully if application fails during state application
            self._reset_state_for_new_spectrum(None)
            # Update UI again after full reset
            self._update_ui_from_current_state()


    def _restore_window_settings(self, state: Dict[str, Any]):
        """Restores window geometry and state from session data."""
        log.debug("Session: Restoring window geometry and state...")
        # Use saved methods from _load_persistent_settings, adapted for dict input
        geom_b64 = state.get('window_geometry')
        if geom_b64 and isinstance(geom_b64, str):
             try: self.restoreGeometry(QByteArray.fromBase64(geom_b64.encode('ascii')))
             except Exception as e: log.error(f"Session: Error restoring geometry: {e}", exc_info=True)
        state_b64 = state.get('window_state')
        if state_b64 and isinstance(state_b64, str):
             try: self.restoreState(QByteArray.fromBase64(state_b64.encode('ascii')))
             except Exception as e: log.error(f"Session: Error restoring state: {e}", exc_info=True)


    def _restore_paths_theme(self, state: Dict[str, Any]):
        """Restores last used paths and theme from session data."""
        log.debug("Session: Restoring paths and theme...")
        default_dir = self._get_default_directory()
        self._last_load_dir = state.get('last_load_dir', default_dir)
        self._last_save_dir = state.get('last_save_dir', default_dir)
        session_theme = state.get('current_theme')
        # Use _load_last_theme logic: validate against available themes, fallback to config/default
        valid_themes = self.theme_manager.get_available_themes()
        config_default = self.config.get('appearance', {}).get('default_theme', DEFAULT_THEME)
        theme_to_apply = config_default # Start with default
        if session_theme and session_theme in valid_themes:
            theme_to_apply = session_theme
        elif session_theme:
             log.warning(f"Session theme '{session_theme}' not found, using default '{config_default}'.")
        # Apply the determined theme
        self.change_theme(theme_to_apply)


    def _reload_spectra_from_session(self, state: Dict[str, Any]) -> Tuple[Optional[Spectrum], List[Spectrum]]:
        """
        Reloads single or multiple spectra based on paths in the session state.

        Returns:
             Tuple containing the reloaded single spectrum (or None) and the list of
             reloaded multiple spectra (or empty list). Returns False if critical reload fails.
        """
        log.debug("Session: Attempting to reload spectra...")
        reloaded_spectrum: Optional[Spectrum] = None
        reloaded_multi_spectra: List[Spectrum] = []
        success = True # Assume success initially

        spectrum_path = state.get('current_spectrum_path')
        multi_paths = state.get('multi_spectra_paths', [])

        # --- Try reloading single spectrum if path exists ---
        if spectrum_path:
            if not os.path.exists(spectrum_path):
                log.error(f"Session: Required spectrum path '{spectrum_path}' not found.")
                # Treat missing single spectrum as critical failure for analysis restore
                success = False # Signal failure
            else:
                # Use saved delimiter/comment if available, else config defaults
                delimiter = state.get('current_spectrum_delimiter') or self.default_delimiter
                comment = state.get('current_spectrum_comment') or self.default_comment_char
                # Handle potential 'None' string saved from repr
                if delimiter == 'None': delimiter = None
                if comment == 'None': comment = None
                try:
                    log.info(f"Session: Reloading single spectrum from {spectrum_path}")
                    reloaded_spectrum = load_spectrum_from_file(spectrum_path, delimiter=delimiter, comment_char=comment)
                except Exception as e:
                    log.error(f"Session: Failed to reload spectrum file '{spectrum_path}': {e}", exc_info=True)
                    # Treat single spectrum reload failure as critical
                    success = False
                    reloaded_spectrum = None

        # --- If no single spectrum, try reloading multiple spectra ---
        elif multi_paths:
            log.info(f"Session contained {len(multi_paths)} multi-spectra paths. Attempting reload...")
            errors = []
            # Use config defaults for multi-load delimiter/comment (assuming they weren't saved per-file)
            delimiter = self.default_delimiter
            comment = self.default_comment_char

            for i, fp in enumerate(multi_paths):
                self.update_status(f"Session: Reloading multi-spectrum {i+1}/{len(multi_paths)}...", 0)
                QApplication.processEvents() # Allow UI update
                if not os.path.exists(fp):
                    error_msg = f"File not found: {os.path.basename(fp)}"
                    errors.append(error_msg)
                    log.warning(f"Session: Multi-spectrum path not found: {fp}")
                    continue # Skip this file
                try:
                    spectrum = load_spectrum_from_file(fp, delimiter=delimiter, comment_char=comment)
                    reloaded_multi_spectra.append(spectrum)
                except Exception as e:
                    error_msg = f"{os.path.basename(fp)}: {e}"
                    errors.append(error_msg)
                    log.warning(f"Session: Failed to reload multi-spectrum file {fp}: {e}")
                    # Continue trying to load others

            log.info(f"Session: Reloaded {len(reloaded_multi_spectra)}/{len(multi_paths)} multi-spectra.")
            if errors:
                 # Log errors but don't necessarily mark the whole load as failed for multi-spectra
                 error_summary = "\n".join([f"- {e}" for e in errors[:5]])
                 if len(errors) > 5: error_summary += f"\n- ... ({len(errors)-5} more)"
                 QMessageBox.warning(self, "Session Load Warning", f"Could not reload all multi-spectra files referenced in session:\n{error_summary}")

        else:
            log.info("Session: No single spectrum path or multi-spectra paths found.")

        # Return success flag along with reloaded data
        if not success:
             QMessageBox.critical(self, "Session Load Error", "Failed to reload the primary spectrum file referenced in the session. Analysis results cannot be restored.")
        return reloaded_spectrum, reloaded_multi_spectra


    def _restore_panel_settings(self, state: Dict[str, Any]):
        """Restores settings for each panel from session data."""
        log.debug("Session: Restoring panel settings...")
        # Map PanelKey Enum to the panel attribute name (more robust than string)
        panel_map = {
            PanelKey.PROCESSING: self.processing_panel,
            PanelKey.DETECTION: self.peak_detection_panel,
            PanelKey.FITTING: self.peak_fitting_panel,
            PanelKey.NIST_SEARCH: self.nist_search_view,
            PanelKey.BOLTZMANN: self.boltzmann_view,
            PanelKey.CFLIBS: self.cf_libs_view,
            PanelKey.ML_ANALYSIS: self.ml_view,
        }

        for panel_key_enum, panel_widget in panel_map.items():
            settings_key = panel_key_enum.value # Get string key like "processing_settings"
            settings = state.get(settings_key)
            if settings is not None and panel_widget:
                if hasattr(panel_widget, 'set_settings') and callable(panel_widget.set_settings):
                    try:
                        panel_widget.set_settings(settings)
                        log.debug(f"Session: Restored settings for panel: {panel_key_enum.name}")
                    except Exception as e:
                        log.error(f"Session: Error restoring settings for panel '{panel_key_enum.name}': {e}", exc_info=True)
                else:
                    log.warning(f"Session: Panel '{panel_key_enum.name}' missing 'set_settings' method.")
            elif settings is not None:
                log.warning(f"Session: Settings found for panel '{panel_key_enum.name}', but panel widget not initialized.")


    def _restore_analysis_results(self, state: Dict[str, Any]):
        """Restores peaks, NIST matches, and plasma parameters from session data."""
        log.debug("Session: Restoring analysis results...")

        # --- Restore Peaks (using Peak.from_dict) ---
        restored_peaks: List[Peak] = []
        peak_data_list = state.get('detected_peaks')
        if isinstance(peak_data_list, list):
            log.info(f"Session: Attempting to restore {len(peak_data_list)} peaks...")
            # Check if from_dict method exists (important after data_models refactor)
            if hasattr(Peak, 'from_dict') and callable(Peak.from_dict):
                valid_peaks = 0
                for i, peak_data in enumerate(peak_data_list):
                    if not isinstance(peak_data, dict): # Basic type check
                        log.warning(f"Session: Skipping invalid peak data (not a dict) at index {i}.")
                        continue
                    try:
                        p = Peak.from_dict(peak_data) # Use the class method
                        if p:
                            restored_peaks.append(p)
                            valid_peaks += 1
                        else:
                            log.warning(f"Session: Skipped invalid peak data (from_dict returned None) at index {i}.")
                    except Exception as e_peak: # Catch errors during individual deserialization
                        log.warning(f"Session: Error deserializing peak data at index {i}: {e_peak}", exc_info=False) # Keep log concise
                self.detected_peaks = restored_peaks
                log.info(f"Session: Successfully restored {valid_peaks}/{len(peak_data_list)} peak objects.")
            else:
                log.error("Session Error: Peak.from_dict method is missing. Cannot restore peaks.")
                self.detected_peaks = []
        else:
             log.debug("Session: No 'detected_peaks' list found in session state.")
             self.detected_peaks = []

        # --- Restore NIST Matches (using NISTMatch.from_dict) ---
        restored_matches: List[NISTMatch] = []
        match_data_list = state.get('nist_matches')
        if isinstance(match_data_list, list):
            log.info(f"Session: Attempting to restore {len(match_data_list)} NIST matches...")
            if hasattr(NISTMatch, 'from_dict') and callable(NISTMatch.from_dict):
                valid_matches = 0
                for i, match_data in enumerate(match_data_list):
                    if not isinstance(match_data, dict):
                         log.warning(f"Session: Skipping invalid NIST match data (not a dict) at index {i}.")
                         continue
                    try:
                        match = NISTMatch.from_dict(match_data) # Use the class method
                        if match:
                            restored_matches.append(match)
                            valid_matches += 1
                        else:
                            log.warning(f"Session: Skipped invalid NIST match data (from_dict returned None) at index {i}.")
                    except Exception as e_match:
                        log.warning(f"Session: Error deserializing NIST match data at index {i}: {e_match}", exc_info=False)
                self.nist_matches = restored_matches
                log.info(f"Session: Successfully restored {valid_matches}/{len(match_data_list)} NIST Match objects.")
                # Re-correlate after restoring both peaks and matches
                self._perform_nist_correlation() # Use the helper method
            else:
                log.error("Session Error: NISTMatch.from_dict method is missing. Cannot restore matches.")
                self.nist_matches = []
        else:
             log.debug("Session: No 'nist_matches' list found in session state.")
             self.nist_matches = []


        # --- Restore Plasma Parameters (handling potential Infinity strings) ---
        temp_val = state.get('plasma_temp_k')
        ne_val = state.get('electron_density_cm3')

        def restore_float_or_inf(value: Any) -> Optional[float]:
            if value is None: return None
            if isinstance(value, str):
                 if value == "Infinity": return np.inf
                 if value == "-Infinity": return -np.inf
                 # Try converting string to float otherwise
                 try: value = float(value)
                 except ValueError: return None # Failed conversion
            # Check if numeric after potential conversion
            if isinstance(value, (int, float)) and np.isfinite(value): return float(value)
            if value == np.inf or value == -np.inf: return float(value) # Allow Inf
            return None # Invalid type or NaN

        self.plasma_temp_k = restore_float_or_inf(temp_val)
        self.electron_density_cm3 = restore_float_or_inf(ne_val)
        log.debug(f"Session: Restored T={self.plasma_temp_k} K, Ne={self.electron_density_cm3} cm⁻³ (handled Infinity strings)")

        # --- Restore DataFrames (using pd.DataFrame.from_records) ---
        self.boltzmann_plot_data = None
        boltzmann_data_list = state.get('boltzmann_plot_data') # Key matches save format now
        if isinstance(boltzmann_data_list, list): # Check if it's a list (saved from to_dict)
            try:
                if boltzmann_data_list: # Only create if list is not empty
                    self.boltzmann_plot_data = pd.DataFrame.from_records(boltzmann_data_list)
                    log.debug(f"Session: Restored Boltzmann plot data DataFrame ({len(self.boltzmann_plot_data)} points).")
                else: # Handle empty list case
                     self.boltzmann_plot_data = pd.DataFrame() # Create empty DataFrame
                     log.debug("Session: Restored empty Boltzmann plot data.")
            except Exception as e:
                log.error(f"Session: Failed to restore Boltzmann plot data from list of records: {e}", exc_info=True)

        self.cf_libs_concentrations = None
        conc_data_list = state.get('cf_libs_concentrations') # Key matches save format now
        if isinstance(conc_data_list, list):
            try:
                if conc_data_list:
                     self.cf_libs_concentrations = pd.DataFrame.from_records(conc_data_list)
                     log.debug(f"Session: Restored CF-LIBS concentrations DataFrame ({len(self.cf_libs_concentrations)} elements).")
                else:
                     self.cf_libs_concentrations = pd.DataFrame()
                     log.debug("Session: Restored empty CF-LIBS concentrations data.")
            except Exception as e:
                log.error(f"Session: Failed to restore CF-LIBS concentrations from list of records: {e}", exc_info=True)


    def _update_ui_from_session(self):
        """DEPRECATED - Use _update_ui_from_current_state instead."""
        # This method is no longer needed as _apply_loaded_session_state now calls
        # the master _update_ui_from_current_state at the end.
        log.warning("_update_ui_from_session is deprecated. Call _update_ui_from_current_state.")
        # self._update_ui_from_current_state()


    # --- Action Handler / Slot Implementations ---

    @pyqtSlot()
    def load_spectrum_action(self):
        """Handles the Load Single Spectrum action."""
        # --- Prompt to save if dirty ---
        if not self._check_save_before_proceeding("load a new spectrum"):
             return # User cancelled

        if self._is_busy:
            log.warning("Load Spectrum action ignored while busy.")
            return
        log.info("Triggered Load Spectrum action.")
        self.update_status("Opening file dialog...")

        file_filter = "Data Files (*.txt *.csv *.asc *.xy);;All Files (*)" # Added *.xy
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Spectrum File", self._last_load_dir, file_filter
        )

        if not filepath:
            log.info("Load Spectrum action cancelled by user.")
            self.update_status("Load cancelled.", 3000)
            return

        self._last_load_dir = os.path.dirname(filepath)
        # Disable critical UI during load
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, f"Loading {os.path.basename(filepath)}...", disable_widgets=widgets_to_disable)
        spectrum = None
        try:
            # Pass config defaults directly to loader
            spectrum = load_spectrum_from_file(
                filepath,
                delimiter=self.default_delimiter, # Pass None to trigger guessing
                comment_char=self.default_comment_char
            )
            # Reset state completely and load the new spectrum
            self._reset_state_for_new_spectrum(spectrum) # This calls master UI update
            # Mark as dirty because new data loaded
            self._mark_dirty(True)
        except (FileNotFoundError, IOError, ValueError, IndexError, Exception) as e:
            # Handle specific load errors
            self._handle_load_error(filepath, e)
            # State is reset within _handle_load_error
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)


    @pyqtSlot()
    def _load_multiple_spectra_action(self):
        """Handles the Load Multiple Spectra action."""
        # --- Prompt to save if dirty ---
        if not self._check_save_before_proceeding("load multiple spectra"):
             return # User cancelled

        if self._is_busy:
            log.warning("Load Multiple Spectra action ignored while busy.")
            return
        log.info("Triggered Load Multiple Spectra action.")
        self.update_status("Opening file dialog for multiple spectra...")

        file_filter = "Data Files (*.txt *.csv *.asc *.xy);;All Files (*)" # Added *.xy
        filepaths, _ = QFileDialog.getOpenFileNames(
            self, "Load Multiple Spectra Files", self._last_load_dir, file_filter
        )

        if not filepaths:
            log.info("Load Multiple Spectra action cancelled by user.")
            self.update_status("Load cancelled.", 3000)
            return

        self._last_load_dir = os.path.dirname(filepaths[0])
        # Disable critical UI during load
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, f"Loading {len(filepaths)} spectra...", disable_widgets=widgets_to_disable)

        loaded_spectra: List[Spectrum] = []
        errors: List[str] = []
        # Use config defaults for loading multiple files
        delimiter = self.config.get('file_io', {}).get('default_delimiter', None) # Allow guessing for multi too
        comment = self.config.get('file_io', {}).get('default_comment_char', '#')

        try:
            for i, fp in enumerate(filepaths):
                # Check for cancellation request? Maybe not needed for file dialog.
                self.update_status(f"Loading {i+1}/{len(filepaths)}: {os.path.basename(fp)}...", 0)
                QApplication.processEvents() # Keep UI responsive during multi-load
                try:
                    spectrum = load_spectrum_from_file(fp, delimiter=delimiter, comment_char=comment)
                    loaded_spectra.append(spectrum)
                except Exception as e:
                    error_msg = f"'{os.path.basename(fp)}': {e}"
                    errors.append(error_msg)
                    log.warning(f"Failed to load spectrum file {fp}: {e}", exc_info=False) # Keep log cleaner

            # --- Update State and UI ---
            num_loaded = len(loaded_spectra)
            num_attempted = len(filepaths)
            status_msg = f"Loaded {num_loaded}/{num_attempted} spectra for ML."
            if errors: status_msg += " Some errors occurred."

            if errors: # Show summary if errors occurred
                error_summary = "\n".join([f"- {e}" for e in errors[:10]])
                if len(errors) > 10: error_summary += f"\n- ... ({len(errors)-10} more)"
                QMessageBox.warning(self, "Load Issues", f"Could not load all files:\n{error_summary}\n\nSee logs for full details.")

            # Set multi_spectra state *before* resetting
            self.multi_spectra = loaded_spectra
            self._reset_state_for_new_spectrum(None) # Clear single spectrum, reset panels

            # Pass loaded spectra to ML view
            if self.ml_view and hasattr(self.ml_view, 'set_spectra_list'):
                self.ml_view.set_spectra_list(loaded_spectra)

            # Mark as dirty because new data loaded
            self._mark_dirty(True)
            # Update status after reset which calls UI update
            self.update_status(status_msg, 5000)

        except Exception as e: # Catch unexpected errors in the loop logic itself
            log.error(f"Critical error during multi-spectrum load loop: {e}", exc_info=True)
            QMessageBox.critical(self, "Load Error", f"An unexpected error occurred during loading:\n{e}")
            self.multi_spectra = [] # Clear partial results
            if self.ml_view and hasattr(self.ml_view, 'set_spectra_list'): self.ml_view.set_spectra_list([])
            self._reset_state_for_new_spectrum(None) # Reset fully on critical error
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)


    # --- Core Processing/Analysis Slots ---

    @pyqtSlot(dict)
    def handle_process_request(self, settings: dict):
        """Handles baseline subtraction, denoising, and smoothing requests."""
        if self.current_spectrum is None:
            QMessageBox.warning(self, "Processing Error", "Please load a single spectrum before processing.")
            return
        if self._is_busy: log.warning("Process request ignored: Busy."); return

        log.info(f"Handling process request with settings: {settings}")
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, "Applying processing steps...", disable_widgets=widgets_to_disable)
        success = False # Track if processing completed without error

        try:
            if self.current_spectrum.raw_intensity is None:
                 raise ValueError("Spectrum is missing raw intensity data.")

            wavelengths = self.current_spectrum.wavelengths
            intensity = self.current_spectrum.raw_intensity.copy() # Start from raw
            processed_intensity = intensity # Initialize
            baseline = np.zeros_like(intensity) # Initialize baseline
            final_baseline = None # Baseline to store in spectrum object

            # --- 1. Baseline Correction ---
            baseline_method = settings.get('baseline_method', 'None')
            if baseline_method == 'Polynomial':
                poly_settings = {k.replace('baseline_poly_', ''): v for k, v in settings.items() if k.startswith('baseline_poly_')}
                log.debug(f"Applying Polynomial Baseline: {poly_settings}")
                processed_intensity, baseline = baseline_poly(wavelengths, intensity, **poly_settings)
                final_baseline = baseline # Store calculated baseline
            elif baseline_method == 'SNIP':
                snip_settings = {k.replace('baseline_snip_', ''): v for k, v in settings.items() if k.startswith('baseline_snip_')}
                log.debug(f"Applying SNIP Baseline: {snip_settings}")
                processed_intensity, baseline = baseline_snip(wavelengths, intensity, **snip_settings)
                final_baseline = baseline
            elif baseline_method != 'None':
                 log.warning(f"Unknown baseline method '{baseline_method}'. Skipping.")
            # If 'None', processed_intensity remains raw_intensity, final_baseline remains None

            # --- 2. Denoising (applied to current processed_intensity) ---
            denoising_method = settings.get('denoising_method', 'None')
            if denoising_method == 'Wavelet':
                wavelet_settings = {k.replace('wavelet_', ''): v for k, v in settings.items() if k.startswith('wavelet_')}
                log.debug(f"Applying Wavelet Denoising: {wavelet_settings}")
                processed_intensity = denoise_wavelet(processed_intensity, **wavelet_settings)
            elif denoising_method != 'None':
                 log.warning(f"Unknown denoising method '{denoising_method}'. Skipping.")

            # --- 3. Smoothing (applied AFTER baseline & denoising) ---
            smoothing_method = settings.get('smoothing_method', 'None')
            if smoothing_method == 'SavitzkyGolay':
                sg_settings = {k.replace('sg_', ''): v for k, v in settings.items() if k.startswith('sg_')}
                log.debug(f"Applying Savitzky-Golay Smoothing: {sg_settings}")
                # This now raises ValueError on invalid params
                processed_intensity = smooth_savitzky_golay(processed_intensity, **sg_settings)
            elif smoothing_method != 'None':
                 log.warning(f"Unknown smoothing method '{smoothing_method}'. Skipping.")

            # --- 4. Update Spectrum Object ---
            self.current_spectrum.update_processed(processed_intensity, final_baseline)
            log.info("Processing complete. Spectrum object updated.")
            self.update_status("Processing complete.", 5000)
            success = True # Mark success

            # --- 5. Clear Downstream Analysis Results ---
            # Clear peaks, which implies clearing fits, nist, plasma
            self._clear_downstream_analysis_data(clear_peaks=True)
            self._mark_dirty(True) # Mark changes as unsaved

        except ValueError as ve: # Catch validation errors (e.g., from SavGol)
             log.error(f"Processing parameter validation error: {ve}")
             QMessageBox.critical(self, "Processing Error", f"Invalid processing parameters:\n{ve}")
             self.update_status("Processing failed (parameter error).", 5000)
             # Don't reset spectrum state if params were just invalid before starting
        except Exception as e:
            log.error(f"Error during processing execution: {e}", exc_info=True)
            QMessageBox.critical(self, "Processing Error", f"An error occurred during processing:\n{e}")
            self.update_status("Processing failed.", 5000)
            # Reset spectrum state on execution error
            if self.current_spectrum: self.current_spectrum.update_processed(None, None)
            self._clear_downstream_analysis_data(clear_peaks=True) # Clear all downstream
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)
            # Update UI regardless of success/failure to reflect current state
            self._update_ui_from_current_state()


    @pyqtSlot(dict)
    def handle_peak_detection_request(self, settings: dict):
        """Handles the peak detection request."""
        if self.current_spectrum is None or self.current_spectrum.processed_intensity is None:
             QMessageBox.warning(self, "Peak Detection", "Please load and process a single spectrum first.")
             return
        if self._is_busy: log.warning("Peak detection request ignored: Busy."); return

        log.info(f"Handling peak detection request with settings: {settings}")
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, "Detecting peaks...", disable_widgets=widgets_to_disable)
        original_peaks = self.detected_peaks # Store in case of failure
        success = False

        try:
            detection_settings = settings.copy()
            method = detection_settings.pop('method', 'Unknown')

            if method == 'ScipyFindPeaks':
                log.debug(f"Using ScipyFindPeaks with params: {detection_settings}")
                # Core function now handles interpolation
                self.detected_peaks = detect_peaks_scipy(self.current_spectrum, **detection_settings)
            # elif method == 'NISTGuided': # Placeholder
            #     # Check if NIST data is available
            #     if not self.nist_matches: # Or maybe needs pre-correlation?
            #          raise ValueError("NIST-Guided detection requires prior NIST search results.")
            #     self.detected_peaks = detect_peaks_nist_guided(self.current_spectrum, self.nist_matches, **detection_settings)
            else:
                raise ValueError(f"Unsupported peak detection method selected: {method}")

            num_peaks = len(self.detected_peaks)
            log.info(f"Peak detection complete. Found {num_peaks} peaks.")
            self.update_status(f"Found {num_peaks} peaks.", 5000)
            success = True
            # Clear downstream results (Fits, NIST, Plasma)
            self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=True) # Keep peaks, clear fits+
            self._mark_dirty(True) # Mark changes

        except ValueError as ve: # Catch config/method errors
            log.error(f"Peak detection configuration error: {ve}")
            QMessageBox.critical(self, "Peak Detection Error", str(ve))
            self.update_status("Peak detection failed (config error).", 5000)
            self.detected_peaks = original_peaks # Revert on failure
        except Exception as e:
            log.error(f"Error during peak detection execution: {e}", exc_info=True)
            QMessageBox.critical(self, "Peak Detection Error", f"An unexpected error occurred:\n{e}")
            self.update_status("Peak detection failed.", 5000)
            self.detected_peaks = original_peaks # Revert on failure
            self._clear_downstream_analysis_data(clear_peaks=True) # Clear everything if detection fails badly
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)
            # Update UI based on final state (new peaks or reverted state)
            self._update_ui_from_current_state()


    @pyqtSlot(dict)
    def handle_peak_fitting_request(self, settings: dict):
        """Handles the request to fit all detected peaks."""
        if not self.detected_peaks:
            QMessageBox.warning(self, "Peak Fitting", "No peaks detected to fit.")
            return
        if self.current_spectrum is None or self.current_spectrum.processed_intensity is None:
             QMessageBox.warning(self, "Peak Fitting", "Processed spectrum data is missing.")
             return
        if self._is_busy: log.warning("Peak fitting request ignored: Busy."); return

        num_peaks_to_fit = len(self.detected_peaks)
        log.info(f"Handling request to fit {num_peaks_to_fit} detected peaks with settings: {settings}")
        # Ensure we have a valid list of profiles
        profiles = settings.get('profiles_to_fit', ['Gaussian', 'Lorentzian', 'PseudoVoigt'])
        if not profiles:
             QMessageBox.warning(self, "Fitting Error", "No fitting profiles specified in settings.")
             return

        # Get min ROI points from config
        min_roi_pts = self.config.get('peak_fitting', {}).get('min_roi_points', 5)
        settings['min_roi_points'] = min_roi_pts # Add to settings dict for fit_peak

        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, f"Fitting {num_peaks_to_fit} peaks...", disable_widgets=widgets_to_disable)
        num_success = 0; num_fail = 0
        # Store current selection to restore later
        selected_peak_idx = self.peak_list_view._selected_list_index if self.peak_list_view else None

        # Use processed intensity for fitting (baseline correction should happen locally in ROI if configured)
        processed_intensity = self.current_spectrum.processed_intensity
        # Create a deep copy of the peak list to store original fits in case of error?
        # original_peaks_state = copy.deepcopy(self.detected_peaks) # Can be memory intensive

        try:
            update_interval = max(1, num_peaks_to_fit // 20) # Update progress roughly every 5%
            for i, peak in enumerate(self.detected_peaks):
                if self._is_busy == False: # Allow cancellation via set_busy(False)? Not standard. Check flag? No simple way here.
                    log.info("Bulk fitting cancelled by user (or external state change).")
                    break # Need a more robust cancellation mechanism if required

                if i % update_interval == 0:
                    self.update_status(f"Fitting peak {i + 1}/{num_peaks_to_fit}...", 0)
                    QApplication.processEvents()

                try:
                    # Pass spectrum, peak index, processed intensity, and fitting settings
                    # fit_peak now returns best_fit, all_results {prof_str: FitResult}
                    best_fit, all_results = fit_peak(
                        spectrum=self.current_spectrum,
                        peak_index=peak.index,
                        processed_intensity=processed_intensity,
                        **settings # Pass fitting parameters
                    )
                    peak.best_fit = best_fit
                    peak.alternative_fits = all_results # Store all attempts

                    if best_fit: num_success += 1
                    else: num_fail += 1
                except Exception as fit_error:
                    log.error(f"Error fitting peak {i} (Index {peak.index}, Wl={peak.wavelength_detected:.3f}): {fit_error}", exc_info=True)
                    num_fail += 1
                    peak.best_fit = None; peak.alternative_fits = {} # Clear fit on error

            log.info(f"Bulk peak fitting complete. Success: {num_success}, Failed: {num_fail}")
            self.update_status(f"Fitting complete: {num_success}/{num_peaks_to_fit} successful.", 5000)
            # Clear downstream results (NIST, Plasma) but keep peaks/fits
            self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=False, clear_nist=True)
            # Re-correlate NIST matches (if any exist) to the new fits
            self._perform_nist_correlation()
            self._mark_dirty(True) # Mark changes

        except Exception as e:
            log.error(f"Critical error during bulk peak fitting loop: {e}", exc_info=True)
            QMessageBox.critical(self, "Peak Fitting Error", f"An unexpected error occurred during fitting:\n{e}")
            self.update_status("Peak fitting failed.", 5000)
            # Revert peak state on critical error? Or clear? Let's clear.
            self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=True) # Clear all fits
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)
            # Update UI based on final state (new fits or cleared fits)
            self._update_ui_from_current_state()
            # Restore selection if possible
            if selected_peak_idx is not None and self.peak_list_view:
                 QTimer.singleShot(0, lambda idx=selected_peak_idx: self.peak_list_view.select_peak_by_index(idx))


    @pyqtSlot(int, dict)
    def handle_refit_single_peak(self, peak_list_index: int, settings: dict):
        """Handles the request to refit a single selected peak."""
        if self._is_busy: log.warning("Refit single peak ignored: Busy."); return

        # --- Validate State ---
        if not (0 <= peak_list_index < len(self.detected_peaks)):
            log.error(f"Invalid peak list index for refit: {peak_list_index} (List size: {len(self.detected_peaks)})")
            QMessageBox.critical(self, "Refit Error", "Internal error: Invalid peak index for refit.")
            return
        peak_to_refit = self.detected_peaks[peak_list_index]
        if self.current_spectrum is None or self.current_spectrum.processed_intensity is None:
            QMessageBox.warning(self, "Refit Error", "Processed spectrum data is missing.")
            return
        # Check if NIST search is running, prevent refit if it modifies correlations?
        if self.nist_search_view and self.nist_search_view._is_searching:
            QMessageBox.warning(self, "Busy", "Cannot refit peak while NIST search is running.")
            return

        log.info(f"Handling refit request for Peak List Index {peak_list_index} (Spectrum Idx {peak_to_refit.index})")
        log.debug(f"Refit settings: {settings}")

        # Add min ROI points from config to settings if not present
        if 'min_roi_points' not in settings:
            settings['min_roi_points'] = self.config.get('peak_fitting', {}).get('min_roi_points', 5)

        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, f"Refitting peak @ {peak_to_refit.wavelength_fitted_or_detected:.2f} nm...", disable_widgets=widgets_to_disable)
        original_fit = copy.deepcopy(peak_to_refit.best_fit) # Deep copy for potential revert
        original_alts = copy.deepcopy(peak_to_refit.alternative_fits)
        success = False

        try:
            best_fit, all_results = fit_peak(
                spectrum=self.current_spectrum,
                peak_index=peak_to_refit.index,
                processed_intensity=self.current_spectrum.processed_intensity,
                **settings
            )
            peak_to_refit.best_fit = best_fit
            peak_to_refit.alternative_fits = all_results
            success = best_fit is not None # Consider refit successful if any fit converged

            if best_fit: log.info(f"Refit successful for peak {peak_list_index}.")
            else: log.warning(f"Refit failed for peak {peak_list_index}.")
            self.update_status(f"Refit {'successful' if best_fit else 'failed'} for peak {peak_list_index}.", 3000)
            # Clear downstream results as a fit changed
            self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=False, clear_nist=True)
            # Re-correlate NIST and mark dirty
            self._perform_nist_correlation()
            self._mark_dirty(True)

        except Exception as e:
            log.error(f"Error refitting peak {peak_list_index} (Index {peak_to_refit.index}): {e}", exc_info=True)
            QMessageBox.critical(self, "Refit Error", f"An error occurred while refitting the peak:\n{e}")
            self.update_status("Refit failed (error).", 5000)
            # Revert to original state on error
            peak_to_refit.best_fit = original_fit
            peak_to_refit.alternative_fits = original_alts
            self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=False, clear_nist=True) # Clear potentially corrupted downstream
            self._perform_nist_correlation() # Recorrelate based on original fit
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)
            # Update UI fully based on potentially changed peak data
            self._update_ui_from_current_state()
            # Ensure the refitted peak remains selected and fitting panel is updated
            if self.peak_list_view: QTimer.singleShot(0, lambda idx=peak_list_index: self.peak_list_view.select_peak_by_index(idx))
            if self.peak_fitting_panel: self.peak_fitting_panel.display_peak_fit_details(peak_to_refit, peak_list_index)


    @pyqtSlot(list) # Expecting List[NISTMatch]
    def _handle_nist_search_results(self, matches: List[NISTMatch]):
        """Handles the raw results received from the NIST search worker thread."""
        if self._is_busy: log.warning("NIST results received but ignored: Application busy."); return

        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        # Set busy during result processing and UI updates
        self.set_busy(True, "Processing NIST results...", disable_widgets=widgets_to_disable)
        try:
            num_matches = len(matches)
            log.info(f"Received {num_matches} potential NIST matches from background search.")
            self.update_status(f"Received {num_matches} NIST matches.", 3000)

            self.nist_matches = matches # Store the raw matches

            # Correlate these matches to the *currently detected and fitted* peaks
            self._perform_nist_correlation() # This updates peak.potential_matches

            # Clear downstream results (Plasma parameters)
            self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=False, clear_plasma=True)
            self._mark_dirty(True) # Mark changes

            # Update UI fully based on new NIST data and correlations
            self._update_ui_from_current_state() # Plots NIST lines, updates tables, updates enablement

        except Exception as e:
             log.error(f"Error handling NIST search results: {e}", exc_info=True)
             QMessageBox.critical(self, "NIST Result Error", f"Failed to process NIST results:\n{e}")
             # Clear potentially corrupted state
             self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=True)
             self._update_ui_from_current_state()
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)

    @pyqtSlot(bool, object, object, object) # success, temp, r2, plot_data
    def _handle_boltzmann_result(self, success: bool, temperature: Optional[float], r_squared: Optional[float], plot_data: Optional[pd.DataFrame]):
        """Handles the results from the Boltzmann calculation (from BoltzmannPlotView)."""
        if self._is_busy: log.warning("Boltzmann result ignored: Application busy."); return

        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, "Processing Boltzmann results...", disable_widgets=widgets_to_disable)
        try:
            log.debug(f"Received Boltzmann result: success={success}, T={temperature}, R²={r_squared}")
            valid_temp = success and isinstance(temperature, (float, int)) and np.isfinite(temperature)

            # Store plot data regardless of success
            self.boltzmann_plot_data = plot_data.copy() if isinstance(plot_data, pd.DataFrame) else None

            if valid_temp:
                # Clear downstream (Ne, Conc) first, then set new temp
                self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=False, clear_plasma=True)
                self.plasma_temp_k = float(temperature)
                r2_str = f"(R²={r_squared:.4f})" if r_squared is not None and np.isfinite(r_squared) else ""
                log.info(f"Boltzmann calculation successful. Stored Tₑ: {self.plasma_temp_k:.2f} K {r2_str}")
                self.update_status(f"Plasma Temperature calculated: {self.plasma_temp_k:.0f} K", 5000)
            else:
                # Calculation failed or returned invalid temperature
                log.warning("Boltzmann calculation failed or returned invalid temperature.")
                self.update_status("Plasma temperature calculation failed.", 5000)
                # Clear downstream (including temp itself)
                self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=False, clear_plasma=True)

            self._mark_dirty(True) # Calculation performed, mark change

        except Exception as e:
             log.error(f"Error handling Boltzmann result: {e}", exc_info=True)
             QMessageBox.critical(self, "Boltzmann Result Error", f"Failed to process Boltzmann results:\n{e}")
             # Clear plasma state on error
             self._clear_downstream_analysis_data(clear_peaks=False, clear_nist=False, clear_plasma=True)
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)
            # Update UI based on new plasma state
            self._update_ui_from_current_state()


    # --- Placeholder Slots for other Calculations (Ne, Conc) ---
    # These should follow a similar pattern: Check busy state, set busy,
    # call core function, handle results/errors, clear downstream, update UI, unset busy.

    @pyqtSlot(str, str, float)
    def handle_ne_calculation_request(self, species1: str, species2: str, temperature_k: float):
        """Handles the request to calculate electron density (Ne)."""
        if self._is_busy: log.warning("Nₑ calculation request ignored: Busy."); return
        # --- Add Prerequisite Checks (Temp, Fits, Correlated NIST) --- #
        if not (self.detected_peaks and any(p.potential_matches for p in self.detected_peaks)):
            QMessageBox.warning(self, "Electron Density", "Requires fitted peaks with NIST correlations."); return
        if not np.isfinite(temperature_k):
            QMessageBox.warning(self, "Electron Density", "Requires a valid plasma temperature."); return

        log.info(f"Handling Nₑ request for {species1}/{species2} at T={temperature_k:.0f} K.")
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, "Calculating Nₑ...", disable_widgets=widgets_to_disable)
        ne_cm3 = None # Initialize
        success = False
        try:
            # TODO: Implement proper filtering of lines_ion1, lines_ion2 from self.detected_peaks based on species
            lines_df1 = pd.DataFrame() # Placeholder
            lines_df2 = pd.DataFrame() # Placeholder
            # Example filtering (needs refinement):
            # lines_df1 = pd.DataFrame([p.best_fit.get_param_dict() | m.to_dict() for p in self.detected_peaks if p.best_fit for m in p.potential_matches if f"{m.element} {m.ion_state_str}" == species1])
            # lines_df2 = pd.DataFrame([...]) # Similarly for species2

            if lines_df1.empty or lines_df2.empty:
                 raise ValueError(f"Could not find suitable fitted/matched lines for species pair {species1}/{species2}.")

            ne_cm3 = calculate_electron_density_saha(lines_df1, lines_df2, temperature_k, species1, species2)
            success = ne_cm3 is not None and np.isfinite(ne_cm3)

            if success:
                # Clear concentrations first, then set Ne
                self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=False, clear_nist=False, clear_plasma=True) # Only clear plasma
                self.electron_density_cm3 = ne_cm3
                self.plasma_temp_k = temperature_k # Ensure temp is stored if Ne calc succeeded
                self.update_status(f"Nₑ calculated: {ne_cm3:.3e} cm⁻³", 5000)
                self._mark_dirty(True)
            else:
                 # Clear Ne and Conc if calculation failed
                 self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=False, clear_nist=False, clear_plasma=True)
                 self.update_status("Nₑ calculation failed or returned invalid value.", 5000)

        except NotImplementedError as nie:
            log.warning(f"Nₑ calculation failed: {nie}")
            QMessageBox.warning(self, "Calculation Not Implemented", str(nie))
            self.update_status(f"Nₑ calculation failed: Not implemented.", 5000)
            self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=False, clear_nist=False, clear_plasma=True)
        except ValueError as ve: # Catch specific errors from calculation or filtering
            log.warning(f"Nₑ calculation failed: {ve}")
            QMessageBox.warning(self, "Electron Density Calculation Error", f"Could not calculate Nₑ:\n{ve}")
            self.update_status(f"Nₑ calculation failed: {ve}", 5000)
            self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=False, clear_nist=False, clear_plasma=True)
        except Exception as e: # Catch unexpected errors
            log.error(f"Error during electron density calculation: {e}", exc_info=True)
            QMessageBox.critical(self, "Calculation Error", f"An unexpected error occurred during Nₑ calculation:\n{e}")
            self.update_status("Nₑ calculation failed (error).", 5000)
            self._clear_downstream_analysis_data(clear_peaks=False, clear_fits=False, clear_nist=False, clear_plasma=True)
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)
            self._update_ui_from_current_state()


    @pyqtSlot(float, object) # temp_k, ne_cm3 (optional, can be None)
    def handle_conc_calculation_request(self, temperature_k: float, ne_cm3_obj: Optional[object]):
        """Handles the request to calculate CF-LIBS concentrations."""
        if self._is_busy: log.warning("Concentration calculation request ignored: Busy."); return
        # --- Prerequisite Checks (Fits, NIST Correlation, Temp) --- #
        if not (self.detected_peaks and any(p.potential_matches for p in self.detected_peaks)):
             QMessageBox.warning(self, "CF-LIBS Calculation", "Requires fitted peaks with NIST correlations."); return
        if not np.isfinite(temperature_k):
             QMessageBox.warning(self, "CF-LIBS Calculation", "Requires a valid plasma temperature."); return

        # Validate Ne input
        ne_cm3: Optional[float] = None
        if ne_cm3_obj is not None:
             try: ne_cm3 = float(ne_cm3_obj);
             except (ValueError, TypeError): pass # Ignore invalid Ne input silently? Or warn?
             if ne_cm3 is not None and not np.isfinite(ne_cm3): ne_cm3 = None # Treat NaN/Inf Ne as None

        log.info(f"Handling CF-LIBS request. T={temperature_k:.0f} K, Nₑ={ne_cm3 if ne_cm3 else 'N/A'}")
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, "Calculating Concentrations...", disable_widgets=widgets_to_disable)
        conc_df = None
        success = False
        try:
            # Core function needs Peaks list (with fits/matches), Temp, optionally Ne
            conc_df = calculate_cf_libs_conc(
                peaks=self.detected_peaks,
                temperature_k=temperature_k,
                electron_density_cm3=ne_cm3 # Pass validated Ne or None
                # Add other params from config if needed (e.g., filtering thresholds)
            )
            success = conc_df is not None # Consider success if DataFrame returned (even if empty?)

            if success:
                 self.cf_libs_concentrations = conc_df
                 num_elements = len(conc_df) if not conc_df.empty else 0
                 log.info(f"CF-LIBS concentrations calculated ({num_elements} elements).")
                 self.update_status(f"CF-LIBS concentrations calculated ({num_elements} elements).", 5000)
                 self._mark_dirty(True)
            else:
                 self.cf_libs_concentrations = None
                 log.warning("CF-LIBS calculation did not return a valid DataFrame.")
                 self.update_status("CF-LIBS calculation failed or returned no results.", 5000)

        except NotImplementedError as nie:
            log.warning(f"CF-LIBS calculation failed: {nie}")
            QMessageBox.warning(self, "Calculation Not Implemented", str(nie))
            self.update_status(f"CF-LIBS calculation failed: Not implemented.", 5000)
            self.cf_libs_concentrations = None
        except ValueError as ve: # Catch specific errors from calculation/filtering
            log.warning(f"CF-LIBS calculation failed: {ve}")
            QMessageBox.warning(self, "CF-LIBS Calculation Error", f"Could not calculate concentrations:\n{ve}")
            self.update_status(f"CF-LIBS calculation failed: {ve}", 5000)
            self.cf_libs_concentrations = None
        except Exception as e: # Catch unexpected errors
            log.error(f"Error during concentration calculation: {e}", exc_info=True)
            QMessageBox.critical(self, "Calculation Error", f"An unexpected error occurred during concentration calculation:\n{e}")
            self.update_status("Concentration calculation failed (error).", 5000)
            self.cf_libs_concentrations = None
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)
            # Update UI based on final state
            self._update_ui_from_current_state()


    # --- UI Interaction Slots ---

    @pyqtSlot(int)
    def handle_peak_selection(self, peak_list_index: int):
        """Handles selection changes from PeakListView or plot clicks."""
        if self._is_busy: return # Ignore UI interaction while busy

        selected_peak: Optional[Peak] = None
        is_valid_selection = False
        if 0 <= peak_list_index < len(self.detected_peaks):
            selected_peak = self.detected_peaks[peak_list_index]
            is_valid_selection = True
            log.debug(f"Peak selection event: List Index {peak_list_index} (Wl={selected_peak.wavelength_fitted_or_detected:.3f})")
        else:
            log.debug(f"Peak selection cleared (index {peak_list_index}).")
            peak_list_index = -1 # Ensure consistent deselection index

        # Update fitting panel with selected peak details
        if self.peak_fitting_panel:
            self.peak_fitting_panel.display_peak_fit_details(selected_peak, peak_list_index if is_valid_selection else None)

        # Highlight peak and its best fit (if any) on the plot
        if self.plot_widget:
            self.plot_widget.highlight_peak(peak_list_index if is_valid_selection else None)
            fit_to_highlight = selected_peak.best_fit if is_valid_selection and selected_peak and selected_peak.best_fit else None
            self.plot_widget.highlight_fit_line(fit_to_highlight)


    @pyqtSlot(object) # FitResult or None
    def handle_show_specific_fit(self, fit_result: Optional[FitResult]):
        """Highlights a specific fit line on the plot (triggered by fitting panel)."""
        if self._is_busy: return
        if self.plot_widget: self.plot_widget.highlight_fit_line(fit_result)


    # --- Internal Data Handling Helpers ---

    def _perform_nist_correlation(self):
        """Internal helper to correlate existing NIST matches with current peaks."""
        if not self.detected_peaks or not self.nist_matches:
             # Clear existing correlations on peaks if no matches or no peaks
             if self.detected_peaks:
                  for peak in self.detected_peaks: peak.potential_matches = []
             log.debug("NIST correlation skipped: No peaks or no matches available.")
             return # Nothing to correlate

        # Get tolerance from NIST search view UI (robustly)
        tolerance_nm = 0.1 # Default tolerance
        try:
            if (self.nist_search_view and hasattr(self.nist_search_view, 'tolerance_dspin') and self.nist_search_view.tolerance_dspin):
                tolerance_nm = self.nist_search_view.tolerance_dspin.value()
            else: log.warning("Could not get tolerance from NIST view UI for correlation. Using default %.2f nm.", tolerance_nm)
        except Exception as e: log.error(f"Error getting tolerance: {e}. Using default %.2f nm.", tolerance_nm, exc_info=True)

        log.info(f"Correlating {len(self.nist_matches)} NIST matches to {len(self.detected_peaks)} peaks (Tol={tolerance_nm:.3f} nm)...")

        # --- Clear existing correlations and perform new ones ---
        for peak in self.detected_peaks:
            peak.potential_matches = [] # Clear previous matches on the peak object

        correlation_count = 0
        unmatched_nist_lines = 0

        for match in self.nist_matches:
            # Basic validation of match object
            if not isinstance(match, NISTMatch) or not hasattr(match, 'wavelength_db') or not np.isfinite(match.wavelength_db):
                 log.debug(f"Skipping invalid NIST match object during correlation: {match!r}")
                 continue
            db_wavelength = match.wavelength_db

            best_peak_match: Optional[Peak] = None
            min_diff = tolerance_nm # Max allowed difference

            # Find the *closest* peak within the tolerance
            for peak in self.detected_peaks:
                peak_wavelength = peak.wavelength_fitted_or_detected
                if not np.isfinite(peak_wavelength): continue # Skip peaks without valid wavelength

                diff = abs(peak_wavelength - db_wavelength)
                if diff <= min_diff: # Found a closer peak within tolerance
                    min_diff = diff
                    best_peak_match = peak

            # Add match to the closest peak found (if any)
            if best_peak_match:
                if hasattr(best_peak_match, 'add_nist_match'):
                     # Add match (add_nist_match handles duplicates and sorting)
                     best_peak_match.add_nist_match(match)
                     correlation_count += 1 # Count successful correlations
                else: log.error(f"Peak object (Index: {best_peak_match.index}) missing 'add_nist_match' method!")
            else:
                unmatched_nist_lines += 1

        log.info(f"Correlation complete. Associated matches to peaks. ({correlation_count} associations, {unmatched_nist_lines} NIST lines remain uncorrelated).")
        # Note: UI update (peak list, plot) happens in the calling function or master update


    # --- Save Actions ---

    SAVE_ACTION_CONFIG: Dict[SaveType, Dict[str, Any]] = {
        # Populated dynamically now based on SaveType Enum
    }

    def _populate_save_config(self):
        """Populates the save action configuration dictionary."""
        # Only run once
        if self.SAVE_ACTION_CONFIG: return

        # Define helper lambda functions (use self inside lambda carefully)
        check_proc = lambda: self.current_spectrum and self.current_spectrum.processed_intensity is not None
        check_peaks = lambda: bool(self.detected_peaks)
        check_nist = lambda: bool(self.nist_matches)
        check_boltz = lambda: self.boltzmann_plot_data is not None and not self.boltzmann_plot_data.empty
        check_conc = lambda: self.cf_libs_concentrations is not None and not self.cf_libs_concentrations.empty
        check_plot = lambda: self.plot_widget and self.plot_widget.figure and (self.plot_widget.ax.lines or self.plot_widget.ax.collections)

        get_spec = lambda: self.current_spectrum
        get_peaks = lambda: self.detected_peaks
        get_nist_df = lambda: pd.DataFrame([m.to_dataframe_row(np.nan, np.nan) for m in self.nist_matches if hasattr(m, 'to_dataframe_row')]) if self.nist_matches else pd.DataFrame()
        get_boltz = lambda: self.boltzmann_plot_data
        get_conc = lambda: self.cf_libs_concentrations
        get_figure = lambda: self.plot_widget.figure if self.plot_widget else None
        # Use lambda for savefig to avoid potential issues with method binding if plot_widget changes
        save_fig_func = lambda fig, filepath, **kwargs: fig.savefig(filepath, **kwargs) if fig else None

        self.SAVE_ACTION_CONFIG = {
            SaveType.PROCESSED_SPECTRUM: {
                "description": "processed spectrum data", "filename_suffix": "_processed.csv",
                "filter": "CSV Files (*.csv);;Text Files (*.txt)", "data_checker": check_proc,
                "data_getter": get_spec, "save_function": save_spectrum_data,
                "save_kwargs": {'include_processed': True} # Ensure processed is included
            },
            SaveType.PEAKS: {
                "description": "peak list", "filename_suffix": "_peaks.csv",
                "filter": "CSV Files (*.csv)", "data_checker": check_peaks,
                "data_getter": get_peaks, "save_function": save_peak_list, "save_kwargs": {}
            },
            SaveType.NIST_MATCHES: {
                "description": "NIST match results", "filename_suffix": "_nist_matches.csv",
                "filter": "CSV Files (*.csv)", "data_checker": check_nist,
                "data_getter": get_nist_df, "save_function": save_dataframe, "save_kwargs": {}
            },
            SaveType.BOLTZMANN_DATA: {
                "description": "Boltzmann plot data", "filename_suffix": "_boltzmann_data.csv",
                "filter": "CSV Files (*.csv)", "data_checker": check_boltz,
                "data_getter": get_boltz, "save_function": save_dataframe, "save_kwargs": {}
            },
            SaveType.CONCENTRATIONS: {
                "description": "CF-LIBS concentrations", "filename_suffix": "_concentrations.csv",
                "filter": "CSV Files (*.csv)", "data_checker": check_conc,
                "data_getter": get_conc, "save_function": save_dataframe, "save_kwargs": {}
            },
            SaveType.PLOT: {
                "description": "main plot image", "filename_suffix": "_spectrum_plot.png",
                "filter": "PNG Image (*.png);;SVG Vector (*.svg);;JPEG Image (*.jpg *.jpeg);;PDF Document (*.pdf);;All Files (*)",
                "data_checker": check_plot, "data_getter": get_figure,
                "save_function": save_fig_func,
                "save_kwargs": {'dpi': 300, 'bbox_inches': 'tight'}
            }
        }

    def _save_action(self, save_type: SaveType):
        """Handles saving different types of data using a configuration-driven approach."""
        # Populate config on first call
        if not self.SAVE_ACTION_CONFIG: self._populate_save_config()

        if self._is_busy: log.warning(f"Save action '{save_type.value}' ignored while busy."); return

        log.info(f"Triggered save action for: {save_type.name}")
        config = self.SAVE_ACTION_CONFIG.get(save_type)
        if not config: log.error(f"Unknown save type: {save_type}"); return

        data_description = config["description"]
        filepath = "" # Initialize filepath to handle potential errors before dialog

        try:
            # 1. Check if data is available
            if not config["data_checker"]():
                log.warning(f"No data available to save for type: {save_type.name}")
                QMessageBox.information(self, "No Data to Save", f"There is no {data_description} available to save.")
                return

            # 2. Get the data to save
            data_to_save = config["data_getter"]()
            # Check again after getting data (e.g., getter might return empty DF or None)
            if data_to_save is None or (isinstance(data_to_save, (pd.DataFrame, list)) and len(data_to_save) == 0):
                 log.warning(f"Data getter for '{save_type.name}' returned None or empty container.")
                 QMessageBox.information(self, "No Data to Save", f"The {data_description} data is currently empty.")
                 return

            # 3. Get Filename from User
            base_name = "libs_forge_output"
            if self.current_spectrum and self.current_spectrum.filename:
                try: base_name = Path(self.current_spectrum.filename).stem
                except Exception: pass
            default_filename = f"{base_name}{config['filename_suffix']}"
            default_path = os.path.join(self._last_save_dir, default_filename)
            file_filter = config["filter"]

            filepath, selected_filter = QFileDialog.getSaveFileName(
                self, f"Save {data_description.capitalize()} As...", default_path, file_filter
            )

            if not filepath: log.info(f"Save action cancelled."); return
            self._last_save_dir = os.path.dirname(filepath)

            # 4. Perform Save Operation (Busy state handled inside)
            self._execute_save(save_type, data_to_save, filepath, config)

        except Exception as e: # Catch errors during data check, getter, or file dialog
            log.error(f"Error preparing save action for '{save_type.name}': {e}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"Could not prepare data or get file path for saving {data_description}:\n{e}")


    def _execute_save(self, save_type: SaveType, data_to_save: Any, filepath: str, config: Dict[str, Any]):
        """Executes the actual save operation within a busy state."""
        widgets_to_disable = self._get_critical_widgets_for_busy_state()
        self.set_busy(True, f"Saving {os.path.basename(filepath)}...", disable_widgets=widgets_to_disable)
        data_description = config["description"]
        try:
            save_function = config["save_function"]
            save_kwargs = config["save_kwargs"]

            # Call the appropriate save function
            # Need to handle different signatures (e.g., savefig vs others)
            if save_type == SaveType.PLOT:
                save_function(data_to_save, filepath, **save_kwargs) # savefig(figure, path, **kwargs)
            else:
                # Assumes other save functions have signature (data, filepath, **kwargs)
                # Or adjust based on specific function needs
                save_function(data_to_save, filepath=filepath, **save_kwargs)

            log.info(f"{data_description.capitalize()} saved successfully to {filepath}")
            self.update_status(f"Saved: {os.path.basename(filepath)}", 5000)
            # Mark clean ONLY if saving the full session? Or per-item save?
            # if save_type == SaveType.SESSION: self._mark_dirty(False)

        except (IOError, PermissionError) as e_io:
            log.error(f"Failed to save {save_type.name} to {filepath}: {e_io}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"Failed to save {data_description} (Permission/IO Error):\n{e_io}")
            self.update_status(f"Save failed: {os.path.basename(filepath)}.", 5000)
        except AttributeError as ae:
             log.error(f"Attribute error during save execution for {save_type.name}: {ae}", exc_info=True)
             QMessageBox.critical(self, "Save Error", f"Failed to save {data_description} (Attribute Error):\n{ae}")
             self.update_status(f"Save failed: {os.path.basename(filepath)}.", 5000)
        except Exception as e:
            log.error(f"Error during save execution for '{save_type.name}': {e}", exc_info=True)
            QMessageBox.critical(self, "Save Error", f"An unexpected error occurred while saving {data_description}:\n{e}")
            self.update_status(f"Save failed: {os.path.basename(filepath)}.", 5000)
        finally:
            self.set_busy(False, disable_widgets=widgets_to_disable)


    # --- External Script Runner ---

    def run_external_script(self, script_relative_path: str, script_args: Optional[List[str]] = None):
        """Runs an external Python script using the ExternalScriptRunnerDialog."""
        if self._is_busy:
            log.warning(f"Request to run script '{script_relative_path}' ignored while busy.")
            QMessageBox.warning(self, "Busy", "Cannot start external script: Application busy.")
            return

        script_args = script_args or []

        try:
            project_root = Path(get_project_root())
            script_absolute_path = (project_root / script_relative_path).resolve() # Ensure absolute path

            if not script_absolute_path.is_file():
                log.error(f"External script not found: {script_absolute_path}")
                QMessageBox.critical(self, "Script Not Found", f"Required script not found:\n{script_absolute_path}")
                return

            # Use sys.executable to ensure same Python environment
            python_executable = sys.executable
            if not python_executable: # Safety check
                 log.error("Could not determine Python executable (sys.executable is empty).")
                 QMessageBox.critical(self, "Python Error", "Cannot determine Python executable path to run the script.")
                 return

            # Format command and arguments for the dialog
            command_str = str(python_executable) # Path to python
            # Arguments list: first arg is the script path, then script's own args
            arguments_list = [str(script_absolute_path)] + script_args
            # Convert list to space-separated string with quotes for display in dialog's line edit
            # Use shlex.join for proper quoting? Or just simple join for display? Let's use simple join.
            arguments_str_display = " ".join(f'"{arg}"' if " " in arg else arg for arg in arguments_list)

            log.info(f"Launching external script runner for: {script_absolute_path}")
            log.debug(f"Command: {command_str}")
            log.debug(f"Arguments passed to script: {script_args}") # Log args passed to script itself

            # Use the dedicated dialog, passing self as parent
            dialog = ExternalScriptRunnerDialog(parent=self)
            # Pre-fill the command and arguments (display representation)
            dialog.set_command(command_str)
            dialog.set_arguments(arguments_str_display) # Display string

            # Execute the dialog modally (blocks MainWindow until closed)
            dialog.exec()
            log.info(f"External script dialog closed for: {script_relative_path}.")

        except ImportError:
             log.error("Failed to import ExternalScriptRunnerDialog.", exc_info=True)
             QMessageBox.critical(self, "Component Error", "Could not find ExternalScriptRunnerDialog.")
        except Exception as e:
            log.error(f"Error setting up/running external script '{script_relative_path}': {e}", exc_info=True)
            QMessageBox.critical(self, "Script Runner Error", f"Could not run external script:\n{e}")


    @pyqtSlot()
    def run_nist_fetcher(self):
        """Runs the NIST data fetching script."""
        self.run_external_script('database/nist_data_fetcher.py')

    @pyqtSlot()
    def run_atomic_data_builder(self):
        """Runs the atomic data building script."""
        QMessageBox.information(self, "Atomic Data Builder",
                                "This script requires user implementation to parse specific atomic level data files.\n\n"
                                "Please ensure you have:\n"
                                "1. Placed your source level data files in 'database/source_atomic_levels/'.\n"
                                "2. **Edited the Python script `database/atomic_data_builder.py`** to correctly parse your file format(s).\n\n"
                                "Click OK to run the script (it will likely show errors if not edited).")
        self.run_external_script('database/atomic_data_builder.py')


    # --- Application Exit ---

    def _check_save_before_proceeding(self, action_description: str) -> bool:
        """Checks if there are unsaved changes and prompts the user to save or cancel."""
        if not self._is_dirty:
             return True # No changes, safe to proceed

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"You have unsaved changes. Save the current session before {action_description}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel # Default button
        )

        if reply == QMessageBox.StandardButton.Save:
            return self._on_save_session_triggered() # Attempt save, return True if successful
        elif reply == QMessageBox.StandardButton.Discard:
            log.info("User chose to discard unsaved changes.")
            return True # Allow proceeding without saving
        else: # Cancel
            log.info(f"User cancelled action '{action_description}' due to unsaved changes.")
            self.update_status(f"{action_description.capitalize()} cancelled.", 3000)
            return False


    def closeEvent(self, event):
        """Handles the main window close event, prompts to save if dirty."""
        log.info("Close event triggered. Checking for unsaved changes...")

        if not self._check_save_before_proceeding("exiting"):
             event.ignore() # Abort the close event if user cancelled
             return

        # If we reach here, user either saved, discarded, or there were no changes.
        log.info("Proceeding with application exit.")

        # --- Stop any running background tasks ---
        log.debug("Stopping background tasks before exit...")
        try:
             if self.nist_search_view and hasattr(self.nist_search_view, '_stop_running_search'):
                  self.nist_search_view._stop_running_search()
             if self.ml_view and hasattr(self.ml_view, '_stop_preprocessing'):
                  self.ml_view._stop_preprocessing()
             # Terminate external QProcess if MainWindow launched one directly
             if self.external_process and self.external_process.state() != QProcess.ProcessState.NotRunning:
                 self.external_process.terminate()
                 if not self.external_process.waitForFinished(500): self.external_process.kill()
                 self.external_process = None
        except Exception as e_stop:
             log.warning(f"Error trying to stop background tasks on exit: {e_stop}")

        # --- Save Persistent Settings ---
        self._save_persistent_settings() # Save geometry, theme etc.

        log.info("Accepting close event. Application will now exit.")
        event.accept() # Allow the window to close