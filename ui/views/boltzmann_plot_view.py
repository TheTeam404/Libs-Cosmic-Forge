# -*- coding: utf-8 -*-
"""
View widget for performing Boltzmann plot analysis and displaying results.

Allows users to:
1. Specify a target species (e.g., "Fe I").
2. Populate a table with candidate spectral lines matching that species.
3. Select which lines to include in the Boltzmann plot.
4. Calculate the electron temperature (Te) based on a linear fit.
5. Visualize the Boltzmann plot and the calculated results.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

# --- PyQt Imports ---
try:
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, QTableWidget,
                                 QHeaderView, QTableWidgetItem, QAbstractItemView, QLabel,
                                 QHBoxLayout, QPushButton, QLineEdit, QMessageBox, QSplitter,
                                 QSizePolicy, QApplication) # Added QApplication
    from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot, QVariant
    from PyQt6.QtGui import QIcon, QBrush, QColor
    _QT_AVAILABLE = True
except ImportError as e:
    _QT_AVAILABLE = False
    logging.critical(f"CRITICAL ERROR in boltzmann_plot_view.py: Cannot import PyQt6 components: {e}.")
    # Cannot function without Qt, re-raise
    raise ImportError(f"PyQt6 components failed to import in boltzmann_plot_view: {e}") from e

# --- Local Imports ---
try:
    from ui.widgets.info_button import InfoButton
    from ui.views.plot_widget import SpectrumPlotWidget
    from core.cflibs import calculate_boltzmann_temp, K_B_EV # Import calculation & constant
except ImportError as e:
    logging.critical(f"CRITICAL ERROR in boltzmann_plot_view.py: Cannot import local dependencies: {e}.")
    raise ImportError(f"Local dependencies failed to import in boltzmann_plot_view: {e}") from e

# --- Constants ---
# Columns expected in the DataFrame passed to display_candidate_lines
REQ_COLS_DISPLAY = {'Peak λ (nm)', 'Intensity', 'Elem', 'Ion', 'DB λ (nm)', 'E_k (eV)', 'g_k', 'A_ki (s⁻¹)'}
# Column names expected by the calculate_boltzmann_temp function (after renaming)
REQ_COLS_CALC = {'intensity', 'wavelength_nm', 'ei_upper', 'gi_upper', 'aki'}
# Mapping from display DataFrame columns to calculation function arguments
RENAME_MAP = {
    'Intensity': 'intensity',
    'Peak λ (nm)': 'wavelength_nm', # Use peak wavelength for intensity term? Or DB lambda? Using peak for now.
    'E_k (eV)': 'ei_upper',       # Upper state energy E_k
    'g_k': 'gi_upper',          # Upper state stat weight g_k
    'A_ki (s⁻¹)': 'aki'         # Transition probability A_ki
}
# Tooltips for table columns
COL_TOOLTIPS = {
    "Use": "Check to include this line in the Boltzmann plot calculation.",
    "Peak λ (nm)": "Fitted or detected wavelength of the peak (nm).",
    "Intensity": "Fitted peak area or amplitude (check CF-LIBS settings).",
    "Elem": "Element symbol from NIST match.",
    "Ion": "Ionization stage from NIST match.",
    "DB λ (nm)": "NIST database wavelength for the matched transition (nm).",
    "E_k (eV)": "Upper energy level (E_k) of the transition from NIST (eV).",
    "g_k": "Statistical weight (g_k) of the upper energy level from NIST.",
    "A_ki (s⁻¹)": "Transition probability (Einstein A coefficient) from NIST (s⁻¹)."
}


class BoltzmannPlotView(QWidget):
    """Displays controls and results for Boltzmann plot temperature calculation."""

    # Signal emitted when the user clicks "Populate Lines"
    populate_lines_requested = pyqtSignal(str) # Carries target species string

    # Signal emitted after calculation attempt
    # Args: success(bool), temperature_k(Optional[float]), r_squared(Optional[float]), plot_data(Optional[pd.DataFrame])
    calculation_complete = pyqtSignal(bool, object, object, object)

    # Minimum number of lines required for a valid fit (configurable?)
    MIN_LINES_FOR_FIT = 3

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config = config.get('cflibs', {}) # Get CF-LIBS sub-config
        # Store DataFrames
        self._candidate_lines_df: Optional[pd.DataFrame] = None # DF passed to display_candidate_lines
        self._plot_data: Optional[pd.DataFrame] = None # DF with x/y coords for plotting
        # Minimum lines needed for fit (get from config or use default)
        self.min_lines_for_fit = self.config.get('min_lines_for_boltzmann', self.MIN_LINES_FOR_FIT)

        if not _QT_AVAILABLE:
             logging.critical("BoltzmannPlotView cannot be initialized: PyQt6 components missing.")
             # Optionally create a placeholder UI indicating the error
             self._init_error_ui("PyQt6 Import Error")
             return

        try:
            self._init_ui()
            self._connect_signals()
            self.calculate_button.setEnabled(False) # Disabled initially
            # self.apply_theme_colors(config) # Apply theme - called by MainWindow typically
        except Exception as e:
             logging.critical(f"Error during BoltzmannPlotView initialization: {e}", exc_info=True)
             self._init_error_ui(f"Initialization Error:\n{e}")

    def _init_error_ui(self, message: str):
         """Creates a simple label indicating an error if UI init fails."""
         layout = QVBoxLayout(self)
         error_label = QLabel(f"Error initializing Boltzmann Plot View:\n{message}\n\nPlease check dependencies and logs.")
         error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
         error_label.setStyleSheet("color: red; font-weight: bold;")
         layout.addWidget(error_label)
         self.setLayout(layout)
         self.setEnabled(False)

    def _init_ui(self):
        """Initializes the UI components."""
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(5)

        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(10)

        # Input Group
        input_group = QGroupBox("1. Setup & Input")
        input_layout = QFormLayout(input_group)
        input_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows); input_layout.setSpacing(8)
        species_hbox = QHBoxLayout(); self.species_input = QLineEdit(); self.species_input.setPlaceholderText("e.g., Fe I, Ca II"); self.species_input.setToolTip("Target species (Element IonState like 'Fe I'). Case-sensitive match needed currently."); species_info_btn = InfoButton(self._show_species_info, "Help on Target Species", self); species_hbox.addWidget(self.species_input, 1); species_hbox.addWidget(species_info_btn); input_layout.addRow("Target Species:", species_hbox)
        self.populate_button = QPushButton("Populate Lines"); self.populate_button.setToolTip("Find identified lines matching the target species with necessary atomic data."); self.populate_button.setIcon(QIcon.fromTheme("edit-find-replace", QIcon.fromTheme("go-down"))); input_layout.addRow(self.populate_button)
        left_layout.addWidget(input_group)

        # Lines Table Group
        lines_group = QGroupBox("2. Select Lines for Plot"); lines_layout = QVBoxLayout(lines_group)
        self.lines_table = QTableWidget()
        self.lines_columns = ["Use", "Peak λ (nm)", "Intensity", "Elem", "Ion", "DB λ (nm)", "E_k (eV)", "g_k", "A_ki (s⁻¹)"]
        self.lines_table.setColumnCount(len(self.lines_columns)); self.lines_table.setHorizontalHeaderLabels(self.lines_columns)
        self.lines_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection); self.lines_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lines_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents); self.lines_table.horizontalHeader().setStretchLastSection(False)
        self.lines_table.verticalHeader().setVisible(False); self.lines_table.setMinimumHeight(180); self.lines_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); self.lines_table.setSortingEnabled(True) # Enable sorting
        lines_layout.addWidget(self.lines_table); left_layout.addWidget(lines_group, 1) # Give table stretch

        # Calculation Group
        calc_group = QGroupBox("3. Calculate Temperature"); calc_layout = QVBoxLayout(calc_group)
        self.calculate_button = QPushButton("Calculate Tₑ"); self.calculate_button.setIcon(QIcon.fromTheme("view-statistics", QIcon.fromTheme("applications-mathematics"))); self.calculate_button.setToolTip(f"Perform Boltzmann plot fit using selected lines (min {self.min_lines_for_fit}).")
        self.result_label = QLabel("Result: Tₑ = N/A"); self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.result_label.setStyleSheet("font-weight: bold; font-size: 10pt; padding: 3px; border: 1px solid gray; border-radius: 3px;")
        calc_layout.addWidget(self.calculate_button); calc_layout.addWidget(self.result_label); left_layout.addWidget(calc_group)

        # --- Right Panel: Boltzmann Plot ---
        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel); right_layout.setContentsMargins(5, 0, 0, 0)
        plot_group = QGroupBox("Boltzmann Plot"); plot_layout = QVBoxLayout(plot_group)
        # Use a SpectrumPlotWidget instance for consistent plot controls and theming
        self.boltzmann_plot_widget = SpectrumPlotWidget(config=self.config, parent=self) # Pass config/parent if needed by plot widget
        # Set specific labels for Boltzmann plot
        self.boltzmann_plot_widget.ax.set_xlabel("Upper Energy Level E$_k$ (eV)")
        self.boltzmann_plot_widget.ax.set_ylabel(r"ln( I $\lambda$ / (A$_{ki}$ g$_k$) )") # Boltzmann Y-axis term
        self.boltzmann_plot_widget.ax.set_title("Boltzmann Plot (Select Lines & Calculate)")
        plot_layout.addWidget(self.boltzmann_plot_widget)
        right_layout.addWidget(plot_group)

        # --- Assemble Splitter ---
        main_splitter.addWidget(left_panel); main_splitter.addWidget(right_panel)
        main_splitter.setStretchFactor(0, 1); main_splitter.setStretchFactor(1, 2) # Give plot more space
        outer_layout = QVBoxLayout(self); outer_layout.addWidget(main_splitter); self.setLayout(outer_layout)

    def _connect_signals(self):
        """Connects internal signals and slots."""
        self.populate_button.clicked.connect(self._request_populate_lines)
        self.calculate_button.clicked.connect(self._trigger_calculation)
        # itemChanged signal connection moved to end of display_candidate_lines

    def _show_species_info(self):
        """Displays help message for the target species input."""
        QMessageBox.information(self, "Target Species Help",
            "Enter the element symbol and ionization state for the Boltzmann plot.\n"
            "Format: 'Element IonState' (e.g., 'Fe I', 'Ca II').\n\n"
            "**Important:**\n"
            "- The species name entered here must **exactly match** the 'Species' string associated with the lines from the NIST search (e.g., 'Fe I' will not match lines labeled 'Fe').\n"
            "- The calculation requires multiple identified lines belonging to this *exact* species.\n"
            "- These lines must have valid atomic data (E\u2096, g\u2096, A\u2096\u1d62) from NIST/correlation.\n\n"
            "Click 'Populate Lines' to find suitable lines among detected/fitted/matched peaks.")

    def clear_all(self):
        """Resets the view to its initial state, clearing all inputs and results."""
        logging.debug("Clearing Boltzmann plot view.")
        self.species_input.clear()
        # Disconnect signal temporarily to avoid triggers during clear
        try: self.lines_table.itemChanged.disconnect(self._handle_line_selection_change)
        except TypeError: pass # Raised if not connected
        self.lines_table.setRowCount(0)
        self.lines_table.setSortingEnabled(False) # Disable sorting while empty
        self._candidate_lines_df = None
        self._plot_data = None
        self.result_label.setText("Result: Tₑ = N/A")
        self.calculate_button.setEnabled(False)

        # Reset plot axes and title
        if hasattr(self, 'boltzmann_plot_widget') and self.boltzmann_plot_widget:
            self.boltzmann_plot_widget.clear_plot(redraw=False)
            self.boltzmann_plot_widget.ax.set_xlabel("Upper Energy Level E$_k$ (eV)")
            self.boltzmann_plot_widget.ax.set_ylabel(r"ln( I $\lambda$ / (A$_{ki}$ g$_k$) )")
            self.boltzmann_plot_widget.ax.set_title("Boltzmann Plot")
            self.boltzmann_plot_widget._redraw_canvas() # Redraw after resetting

    @pyqtSlot()
    def _request_populate_lines(self):
        """Validates species input and emits the populate_lines_requested signal."""
        species = self.species_input.text().strip()
        if not species:
            QMessageBox.warning(self, "Missing Input", "Please enter the target species (e.g., 'Fe I').")
            return
        # Basic format check (more robust check might be needed)
        parts = species.split()
        if len(parts) != 2:
            QMessageBox.warning(self, "Invalid Format", "Species format incorrect. Use 'Element IonState' (e.g., 'Fe I', 'Ca II').")
            return

        logging.info(f"Requesting population of candidate lines for species: {species}")
        self.populate_lines_requested.emit(species) # Emit signal for MainWindow


    def display_candidate_lines(self, lines_df: Optional[pd.DataFrame]):
        """
        Populates the lines table with candidate lines found for the target species.
        Sets tooltips for column headers.
        """
        # --- Signal Disconnect ---
        try: self.lines_table.itemChanged.disconnect(self._handle_line_selection_change)
        except TypeError: pass

        self.lines_table.setSortingEnabled(False)
        self.lines_table.setRowCount(0)
        self._candidate_lines_df = lines_df # Store reference
        self._plot_data = None # Clear previous plot calculation data
        if hasattr(self, 'boltzmann_plot_widget'): self.boltzmann_plot_widget.clear_plot() # Clear plot
        self.result_label.setText("Result: Tₑ = N/A")

        if lines_df is None or lines_df.empty:
            logging.info("No candidate lines provided for Boltzmann plot.")
            self.calculate_button.setEnabled(False)
            self.lines_table.setSortingEnabled(True)
            return

        # --- Column Validation ---
        missing_cols = REQ_COLS_DISPLAY - set(lines_df.columns)
        if missing_cols:
             logging.error(f"Input DataFrame for Boltzmann lines missing required columns: {missing_cols}")
             QMessageBox.critical(self, "Data Error", f"Line data missing essential columns:\n{', '.join(missing_cols)}")
             self.calculate_button.setEnabled(False)
             self.lines_table.setSortingEnabled(True)
             return

        logging.info(f"Displaying {len(lines_df)} candidate lines for Boltzmann plot.")
        self.lines_table.setRowCount(len(lines_df))
        # --- Cache column indices ---
        col_map = {name: idx for idx, name in enumerate(self.lines_columns)}

        # --- Populate Table ---
        for row_idx, data_row in lines_df.iterrows():
            # Checkbox column ("Use")
            chk_item = QTableWidgetItem(); chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled); chk_item.setCheckState(Qt.CheckState.Checked); chk_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter); self.lines_table.setItem(row_idx, col_map["Use"], chk_item)
            # Data columns using helper
            self.lines_table.setItem(row_idx, col_map["Peak λ (nm)"], self._create_table_item(data_row.get('Peak λ (nm)'), 4))
            self.lines_table.setItem(row_idx, col_map["Intensity"], self._create_table_item(data_row.get('Intensity'), 1))
            self.lines_table.setItem(row_idx, col_map["Elem"], self._create_table_item(data_row.get('Elem'), alignment=Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter))
            self.lines_table.setItem(row_idx, col_map["Ion"], self._create_table_item(data_row.get('Ion'), alignment=Qt.AlignmentFlag.AlignCenter|Qt.AlignmentFlag.AlignVCenter))
            self.lines_table.setItem(row_idx, col_map["DB λ (nm)"], self._create_table_item(data_row.get('DB λ (nm)'), 4))
            self.lines_table.setItem(row_idx, col_map["E_k (eV)"], self._create_table_item(data_row.get('E_k (eV)'), 4))
            self.lines_table.setItem(row_idx, col_map["g_k"], self._create_table_item(data_row.get('g_k'), 0))
            self.lines_table.setItem(row_idx, col_map["A_ki (s⁻¹)"], self._create_table_item(data_row.get('A_ki (s⁻¹)'), scientific=True))

        # --- Set Header Tooltips ---
        header = self.lines_table.horizontalHeader()
        for i, col_name in enumerate(self.lines_columns):
            tooltip = COL_TOOLTIPS.get(col_name, col_name) # Use name itself as fallback
            header_item = self.lines_table.horizontalHeaderItem(i)
            if header_item: header_item.setToolTip(tooltip)
            else: logging.warning(f"Could not get header item for column index {i} ('{col_name}') to set tooltip.")

        self.lines_table.resizeColumnsToContents()
        self.lines_table.horizontalHeader().setStretchLastSection(True)
        self.lines_table.setSortingEnabled(True)

        # --- Reconnect signal AFTER population ---
        self.lines_table.itemChanged.connect(self._handle_line_selection_change)
        self._handle_line_selection_change() # Trigger initial check


    def _create_table_item(self, value: Any, precision: Optional[int] = None, scientific: bool = False, alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) -> QTableWidgetItem:
        """Helper function to create and format a QTableWidgetItem."""
        item = QTableWidgetItem()
        text = ""
        data_for_sorting = None # Store numeric or string for sorting

        if value is None or (isinstance(value, (float, np.number)) and not np.isfinite(value)):
            text = "" # Display empty for invalid data
            item.setForeground(QBrush(QColor('gray'))) # Gray out invalid data
            # data_for_sorting = None # Keep as None
        elif isinstance(value, (int, np.integer)):
            text = str(value)
            data_for_sorting = int(value)
        elif isinstance(value, (float, np.floating)):
            if scientific: text = f"{value:.2e}"
            elif precision is not None: text = f"{value:.{precision}f}"
            else: text = f"{value:.4g}"
            data_for_sorting = float(value)
            item.setToolTip(str(value)) # Tooltip shows full precision
        else:
            text = str(value) # Assume string
            alignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter # Default left align
            data_for_sorting = text

        item.setText(text)
        item.setTextAlignment(alignment)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable) # Read-only

        # Store data for sorting using DisplayRole (Qt sorts based on this by default)
        if data_for_sorting is not None:
             item.setData(Qt.ItemDataRole.DisplayRole, data_for_sorting)

        return item


    def _handle_line_selection_change(self, item: Optional[QTableWidgetItem] = None):
        """Updates the 'Calculate' button state based on number of selected lines."""
        # item arg is passed by signal but not strictly needed here
        selected_count = 0
        try:
             use_col_idx = self.lines_columns.index("Use")
             for row in range(self.lines_table.rowCount()):
                 chk_item = self.lines_table.item(row, use_col_idx)
                 if chk_item and chk_item.checkState() == Qt.CheckState.Checked:
                     selected_count += 1
        except ValueError: # 'Use' column not found
             logging.error("Could not find 'Use' column in lines table to check selection.")
             self.calculate_button.setEnabled(False)
             return
        except Exception as e:
             logging.error(f"Error checking line selection: {e}", exc_info=True)
             self.calculate_button.setEnabled(False)
             return

        can_calculate = selected_count >= self.min_lines_for_fit
        self.calculate_button.setEnabled(can_calculate)
        logging.debug(f"{selected_count} lines selected for Boltzmann plot (min required: {self.min_lines_for_fit}). Calculate button enabled: {can_calculate}")


    @pyqtSlot()
    def _trigger_calculation(self):
        """Gathers selected line data, triggers calculation, and displays results."""
        if self._candidate_lines_df is None or self._candidate_lines_df.empty:
            logging.warning("Calculation triggered, but no candidate line data available.")
            return

        use_col_idx = self.lines_columns.index("Use")
        # Get indices from the original DataFrame based on checked rows
        selected_df_indices = []
        for r in range(self.lines_table.rowCount()):
             chk_item = self.lines_table.item(r, use_col_idx)
             if chk_item and chk_item.checkState() == Qt.CheckState.Checked:
                 # Assuming table row index corresponds to original DataFrame index after population
                 # This can break if table is sorted AFTER population. Safer to store DF index?
                 # Let's assume no sorting between population and calculation for now.
                 selected_df_indices.append(self.lines_table.visualRow(r)) # Or use stored index if added

        if len(selected_df_indices) < self.min_lines_for_fit:
            QMessageBox.warning(self, "Not Enough Lines", f"Please select at least {self.min_lines_for_fit} lines.")
            return

        # Select rows from the *stored* DataFrame using iloc (integer location)
        selected_df = self._candidate_lines_df.iloc[selected_df_indices].copy()

        # --- Prepare DataFrame for calculation function ---
        missing_cols = set(RENAME_MAP.keys()) - set(selected_df.columns)
        if missing_cols:
            logging.error(f"Cannot calculate Boltzmann T: DataFrame is missing required columns: {missing_cols}")
            QMessageBox.critical(self, "Data Error", f"Cannot perform calculation. Data missing:\n{', '.join(missing_cols)}")
            return

        # Select and rename columns
        calc_df = selected_df[list(RENAME_MAP.keys())].rename(columns=RENAME_MAP)

        # Add label column if source exists
        label_col_source = 'DB λ (nm)' # Use DB wavelength for label
        if label_col_source in selected_df.columns:
             calc_df['label'] = selected_df[label_col_source].apply(lambda x: f"{x:.3f} nm" if pd.notna(x) else "?")
        else: calc_df['label'] = None

        logging.info(f"Calculating Boltzmann temperature using {len(calc_df)} selected lines.")
        # TODO: Refactor calculation to background thread (Issue 18.6)
        self.parent().set_busy(True, "Calculating Boltzmann Temperature...") # Access parent's busy state
        QApplication.processEvents() # Update UI before potentially long calculation

        success = False; temp_k = None; temp_err = None; r_squared = None; plot_data_df = None
        try:
            temp_k, temp_err, r_squared, plot_data_df = calculate_boltzmann_temp(calc_df)
            self._plot_data = plot_data_df # Store data used for plotting
            success = temp_k is not None and np.isfinite(temp_k)
        except ImportError: # Catch SciPy missing error
             logging.error("Boltzmann calculation failed: SciPy is required but not installed.")
             QMessageBox.critical(self, "Dependency Error", "SciPy library is required for Boltzmann plot calculation.\nPlease install it (`pip install scipy`).")
        except Exception as e:
            logging.error(f"Error during Boltzmann calculation: {e}", exc_info=True)
            QMessageBox.critical(self, "Calculation Error", f"An error occurred during calculation:\n{e}")
        finally:
             if hasattr(self.parent(), 'set_busy'): self.parent().set_busy(False) # Reset busy state

        # --- Display Results ---
        if success:
            t_str = f"{temp_k:.0f}"; err_str = f"{temp_err:.0f}" if temp_err is not None and np.isfinite(temp_err) else "N/A"; r2_str = f"{r_squared:.3f}" if r_squared is not None and np.isfinite(r_squared) else "N/A"
            self.result_label.setText(f"Result: Tₑ ≈ {t_str} ± {err_str} K (R² = {r2_str})")
        else:
            r2_str = f"{r_squared:.3f}" if r_squared is not None and np.isfinite(r_squared) else "N/A"
            msg = "Calculation Failed";
            if r_squared is not None: msg += f" (R²={r2_str})"
            self.result_label.setText(f"Result: {msg}")

        # Emit signal with results
        self.calculation_complete.emit(success, temp_k, r_squared, self._plot_data)
        # Update plot (handles None data)
        self._update_boltzmann_plot(self._plot_data, temp_k, r_squared)


    def _update_boltzmann_plot(self, plot_data: Optional[pd.DataFrame], temp_k: Optional[float], r_squared: Optional[float]):
        """Updates the Boltzmann plot with the calculated data and fit line."""
        # Ensure plot widget exists
        if not hasattr(self, 'boltzmann_plot_widget') or not self.boltzmann_plot_widget:
            logging.error("Cannot update Boltzmann plot: Plot widget not initialized.")
            return

        plot_widget = self.boltzmann_plot_widget; ax = plot_widget.ax
        plot_widget.clear_plot(redraw=False)
        ax.set_xlabel("Upper Energy Level E$_k$ (eV)"); ax.set_ylabel(r"ln( I $\lambda$ / (A$_{ki}$ g$_k$) )")
        title = "Boltzmann Plot"

        if plot_data is None or plot_data.empty or 'x_energy_ev' not in plot_data.columns or 'y_boltzmann_term' not in plot_data.columns:
            ax.set_title(title + " (No valid data to plot)")
            logging.debug("Updating Boltzmann plot: No valid data.")
            plot_widget._redraw_canvas()
            return

        x = plot_data['x_energy_ev'].values; y = plot_data['y_boltzmann_term'].values
        finite_mask = np.isfinite(x) & np.isfinite(y); x_plot = x[finite_mask]; y_plot = y[finite_mask]

        if len(x_plot) == 0: ax.set_title(title + " (No finite data points)"); plot_widget._redraw_canvas(); return

        # Plotting (colors should ideally come from theme config via plot_widget)
        scatter_color = 'blue'; fit_color = 'red' # Fallback colors
        scatter = ax.scatter(x_plot, y_plot, marker='o', color=scatter_color, label=f"Data ({len(x_plot)} points)", zorder=5)

        # Add text labels to points (use label column if exists)
        if 'label' in plot_data.columns and len(x_plot) < 25: # Only label if few points
            labels = plot_data['label'].values[finite_mask]
            for i, txt in enumerate(labels):
                 if txt and isinstance(txt, str):
                      ax.text(x_plot[i], y_plot[i], f' {txt}', fontsize=7, va='bottom', ha='left', clip_on=True)

        # Plot fit line if calculation was successful
        fit_line = None
        if temp_k is not None and np.isfinite(temp_k) and len(x_plot) >= 2:
            try:
                slope = -1.0 / (temp_k * K_B_EV); intercept = np.mean(y_plot) - slope * np.mean(x_plot)
                x_line = np.array([np.min(x_plot), np.max(x_plot)]); y_line = intercept + slope * x_line
                fit_line, = ax.plot(x_line, y_line, '--', color=fit_color, lw=1.5, label=f'Fit (R²={r_squared:.3f})' if r_squared is not None else 'Fit', zorder=3)
                title += f": T$_e$ ≈ {temp_k:.0f} K"
            except Exception as e: title += " (Fit Line Error)"; logging.warning(f"Could not plot Boltzmann fit line: {e}")
        elif temp_k is None and r_squared is not None: title += " (Invalid Fit)"

        ax.set_title(title)
        handles = [scatter]; labels = [scatter.get_label()]
        if fit_line: handles.append(fit_line); labels.append(fit_line.get_label())
        plot_widget._update_legend(handles=handles, labels=labels, loc='best')
        plot_widget.apply_theme_colors(self.config) # Apply theme
        plot_widget._redraw_canvas()
        logging.debug("Boltzmann plot updated.")


    def set_restored_data(self, temperature_k: Optional[float], plot_data: Optional[pd.DataFrame]):
         """Applies state loaded from a session file."""
         logging.info("Restoring state to Boltzmann view.")
         self._plot_data = plot_data

         # Update result label based on loaded temp
         success = temperature_k is not None and np.isfinite(temperature_k)
         if success:
             t_str = f"{temperature_k:.0f}"
             self.result_label.setText(f"Result: Tₑ ≈ {t_str} K (Restored)")
             # We don't have R² stored, cannot display it accurately
         else:
             self.result_label.setText("Result: Tₑ = N/A (Restored)")

         # Re-plot the loaded data (R² will be missing from label)
         self._update_boltzmann_plot(self._plot_data, temperature_k, r_squared=None)

         # Emit signal to notify MainWindow about the restored state (important for CF-LIBS)
         # Emit None for R^2 as it wasn't restored directly
         self.calculation_complete.emit(success, temperature_k, None, self._plot_data)
         # Keep calculation button disabled as selection state isn't restored
         self.calculate_button.setEnabled(False)


    def apply_theme_colors(self, config: Dict):
        """Applies color settings from the theme to the plot."""
        if hasattr(self, 'boltzmann_plot_widget') and self.boltzmann_plot_widget:
            self.boltzmann_plot_widget.apply_theme_colors(config) # Delegate