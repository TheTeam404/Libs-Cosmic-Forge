# -*- coding: utf-8 -*-
"""
Utility functions and a dialog for running external scripts/commands.

Contains:
- ExternalScriptRunnerDialog: A PyQt dialog to execute external processes using QProcess,
                               displaying stdout/stderr in real-time.

Note: Utility functions (get_project_root, setup_logging, ensure_odd) previously
      in this file are assumed to be accessible via the 'utils.helpers' module
      if needed elsewhere in the project. They are removed from here to keep
      this file focused on the script runner dialog.
"""

import os
import sys
import logging
import shlex   # For robust argument splitting
import locale  # For getting default system encoding
import subprocess # Underlying mechanism, useful for context

# --- PyQt Imports (Needed for the Dialog) ---
_QT_AVAILABLE = False
try:
    from PyQt6.QtWidgets import (
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
        QPlainTextEdit, QFileDialog, QMessageBox, QDialogButtonBox, QTextEdit, # Added QTextEdit for potential env vars
        QSizePolicy, QApplication, QWidget
    )
    from PyQt6.QtCore import QProcess, Qt, pyqtSlot, QProcessEnvironment, QTimer # Added QProcessEnvironment, QTimer
    from PyQt6.QtGui import QFont, QIcon # Added QIcon
    _QT_AVAILABLE = True
except ImportError:
    # Define dummy classes if PyQt is not available
    class QDialog: pass
    class QProcess: pass
    class QPushButton: pass
    class QPlainTextEdit: pass
    class QLineEdit: pass # Add missing dummies
    class QLabel: pass
    class QVBoxLayout: pass
    class QHBoxLayout: pass
    class QFileDialog: pass
    class QMessageBox: pass
    class QDialogButtonBox: pass
    class QSizePolicy: pass
    class QWidget: pass
    class QProcessEnvironment: pass
    class QTimer: pass
    class QFont: pass
    class QIcon: pass
    def pyqtSlot(*args, **kwargs): return lambda func: func # Dummy decorator
    # Add Qt types if needed
    class Qt:
        class Orientation: Horizontal = 0; Vertical = 1
        class AlignmentFlag: AlignLeft = 0; AlignRight = 0; AlignCenter = 0; AlignVCenter = 0
        class WindowModality: WindowModal = 0; NonModal = 0
        class FocusPolicy: NoFocus = 0
        class ItemDataRole: UserRole = 0; DisplayRole = 0
        class CheckState: Unchecked = 0; Checked = 0
        class ContextMenuPolicy: NoContextMenu = 0
        class LineWrapMode: NoWrap = 0
        class StandardPixmap: SP_MessageBoxQuestion = 0
        class DockWidgetArea: LeftDockWidgetArea=0; RightDockWidgetArea=0; BottomDockWidgetArea=0; AllDockWidgetAreas=0
        class ToolBarArea: TopToolBarArea=0
    # Define pyqtSignal as dummy if needed
    if 'pyqtSignal' not in globals():
         class pyqtSignal:
              def __init__(self, *args, **kwargs): pass
              def connect(self, *args, **kwargs): pass
              def disconnect(self, *args, **kwargs): pass
              def emit(self, *args, **kwargs): pass


    logging.warning("PyQt6 not found. ExternalScriptRunnerDialog will not be available.")


# --- External Script Runner Dialog ---

# Only define the class if PyQt was imported successfully
if _QT_AVAILABLE:
    class ExternalScriptRunnerDialog(QDialog):
        """
        A dialog to execute external scripts or commands using QProcess,
        displaying stdout and stderr. Handles basic argument parsing and
        offers optional environment variable configuration.
        """
        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.setWindowTitle("Run External Script/Command")
            self.setMinimumSize(650, 450) # Slightly larger default size

            # --- QProcess Setup ---
            self.process = QProcess(self)
            # Connect signals to appropriate slots
            self.process.readyReadStandardOutput.connect(self._handle_stdout)
            self.process.readyReadStandardError.connect(self._handle_stderr)
            self.process.stateChanged.connect(self._handle_state_change)
            self.process.finished.connect(self._handle_finished) # Handles exit code/status
            self.process.errorOccurred.connect(self._handle_error) # Handles launch/process errors

            # --- UI Initialization ---
            self._init_ui()
            # Set initial button states based on NotRunning state
            self._update_button_states(QProcess.ProcessState.NotRunning)

        def _init_ui(self):
            """Initialize UI elements and layout."""
            layout = QVBoxLayout(self)

            # --- Command Input ---
            cmd_layout = QHBoxLayout()
            cmd_layout.addWidget(QLabel("Command/Script:"))
            self.command_input = QLineEdit()
            self.command_input.setPlaceholderText("Enter command or full path to script/executable")
            cmd_layout.addWidget(self.command_input)
            self.browse_button = QPushButton("Browse...")
            self.browse_button.setToolTip("Browse for an executable or script")
            self.browse_button.clicked.connect(self._browse_script)
            cmd_layout.addWidget(self.browse_button)
            layout.addLayout(cmd_layout)

            # --- Arguments Input ---
            args_layout = QHBoxLayout()
            args_layout.addWidget(QLabel("Arguments:"))
            self.args_input = QLineEdit()
            self.args_input.setPlaceholderText("Enter arguments (use quotes for spaces if needed)")
            self.args_input.setToolTip("Arguments passed to the command. Use quotes for arguments containing spaces.")
            args_layout.addWidget(self.args_input)
            layout.addLayout(args_layout)

            # --- Optional: Environment Variables (Commented out by default) ---
            # self.env_button = QPushButton("Environment Variables...")
            # self.env_button.setCheckable(True)
            # self.env_button.toggled.connect(self._toggle_env_vars)
            # args_layout.addWidget(self.env_button) # Add to args layout
            #
            # self.env_vars_edit = QTextEdit()
            # self.env_vars_edit.setPlaceholderText("Enter environment variables (e.g., KEY=VALUE), one per line.")
            # self.env_vars_edit.setToolTip("Define custom environment variables for the process.\nFormat: VARIABLE_NAME=variable_value\nOne definition per line.")
            # self.env_vars_edit.setVisible(False) # Initially hidden
            # self.env_vars_edit.setFixedHeight(80) # Limit height
            # layout.addWidget(self.env_vars_edit)
            # --- End Optional Environment Variables ---

            # --- Output Area ---
            layout.addWidget(QLabel("Output:"))
            self.output_area = QPlainTextEdit()
            self.output_area.setReadOnly(True)
            self.output_area.setFont(QFont("Courier New", 9)) # Monospaced font
            self.output_area.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap) # Keep lines long
            layout.addWidget(self.output_area)

            # --- Status Label ---
            self.status_label = QLabel("Status: Ready")
            self.status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            layout.addWidget(self.status_label)

            # --- Control Buttons ---
            self.button_box = QDialogButtonBox()
            # Standard roles for better platform integration
            self.run_button = self.button_box.addButton("Run", QDialogButtonBox.ButtonRole.ActionRole)
            self.stop_button = self.button_box.addButton("Stop", QDialogButtonBox.ButtonRole.ActionRole)
            self.close_button = self.button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)

            # Set icons if desired (using theme icons as example)
            self.run_button.setIcon(QIcon.fromTheme("system-run"))
            self.stop_button.setIcon(QIcon.fromTheme("process-stop"))
            self.close_button.setIcon(QIcon.fromTheme("window-close"))

            self.run_button.clicked.connect(self.start_script)
            self.stop_button.clicked.connect(self.stop_script)
            self.close_button.clicked.connect(self.reject) # Default reject action closes dialog

            layout.addWidget(self.button_box)
            self.setLayout(layout)

        # --- Public Methods to Pre-fill ---
        def set_command(self, command: str):
            """Sets the text in the command input field."""
            self.command_input.setText(command)

        def set_arguments(self, args_str: str):
            """Sets the text in the arguments input field."""
            self.args_input.setText(args_str)

        # --- Optional Environment Variable UI ---
        # @pyqtSlot(bool)
        # def _toggle_env_vars(self, checked):
        #     """Shows/hides the environment variable input area."""
        #     self.env_vars_edit.setVisible(checked)

        # def _get_environment(self) -> Optional[QProcessEnvironment]:
        #     """Parses environment variables from the text edit."""
        #     if not self.env_vars_edit.isVisible() or not self.env_vars_edit.toPlainText().strip():
        #         return None # Use default environment if not visible or empty
        #
        #     env = QProcessEnvironment.systemEnvironment() # Start with system env
        #     text = self.env_vars_edit.toPlainText()
        #     lines = text.splitlines()
        #     parse_errors = []
        #     for i, line in enumerate(lines):
        #         line = line.strip()
        #         if not line or line.startswith('#'): # Skip empty/comment lines
        #             continue
        #         if '=' not in line:
        #             parse_errors.append(f"Line {i+1}: Missing '=' separator ('{line}')")
        #             continue
        #         key, value = line.split('=', 1)
        #         key = key.strip()
        #         value = value.strip() # Keep value as is (user handles quotes if needed)
        #         if not key:
        #             parse_errors.append(f"Line {i+1}: Variable name cannot be empty.")
        #             continue
        #         logging.debug(f"Setting environment variable: {key}={value}")
        #         env.insert(key, value)
        #
        #     if parse_errors:
        #          QMessageBox.warning(self, "Environment Variable Error",
        #                              "Errors parsing environment variables:\n- " + "\n- ".join(parse_errors))
        #          return None # Indicate error by returning None? Or return partial env? Let's return None.
        #
        #     return env

        # --- Internal Slots and Helpers ---

        @pyqtSlot()
        def _browse_script(self):
            """Open a file dialog to select a script/executable."""
            # Use user's home directory or last used directory as starting point
            start_dir = getattr(self, '_last_browse_dir', str(Path.home()))
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Script or Executable", start_dir,
                "All Files (*);;Python Scripts (*.py);;Batch Files (*.bat *.cmd);;Shell Scripts (*.sh)"
            )
            if file_path:
                self.command_input.setText(file_path)
                self._last_browse_dir = os.path.dirname(file_path) # Remember directory

        @pyqtSlot()
        def start_script(self):
            """Starts the external process using QProcess."""
            if self.process.state() != QProcess.ProcessState.NotRunning:
                 logging.warning("Start script requested, but process is already running or starting.")
                 return

            command = self.command_input.text().strip()
            if not command:
                QMessageBox.warning(self, "Input Error", "Please enter a command or select a script path.")
                return

            args_str = self.args_input.text().strip()
            arguments: List[str] = []
            try:
                # Use shlex for robust splitting, respecting quotes
                # Use posix=False on Windows to handle backslashes in paths correctly
                arguments = shlex.split(args_str, posix=(os.name != 'nt'))
            except ValueError as e_shlex:
                 QMessageBox.warning(self, "Argument Error", f"Could not parse arguments (check quoting):\n{e_shlex}")
                 return

            self.output_area.clear() # Clear previous output
            self.status_label.setText("Status: Starting...")
            self._update_button_states(QProcess.ProcessState.Starting) # Update buttons immediately
            QApplication.processEvents() # Ensure UI updates

            program = command
            program_args = arguments

            # --- Handle Python Scripts ---
            # If command ends with .py and isn't explicitly 'python' or 'python3', prepend sys.executable
            if command.lower().endswith(".py") and not os.path.basename(command).lower().startswith("python"):
                 executable = sys.executable # Use the same python interpreter that runs the GUI app
                 logging.info(f"Prepending Python executable ('{executable}') to run script: '{command}'")
                 program_args.insert(0, program) # Script path becomes the first argument
                 program = executable # Executable is now python

            logging.info(f"Running command: '{program}' with arguments: {program_args}")

            # --- Get Optional Environment --- (Uncomment if using Env Vars UI)
            # process_environment = self._get_environment()
            # if process_environment is None and self.env_vars_edit.isVisible():
            #      # Error occurred parsing environment variables, stop execution
            #      self.status_label.setText("Status: Error (Environment)")
            #      self._update_button_states(QProcess.ProcessState.NotRunning)
            #      return
            # elif process_environment:
            #      logging.debug("Setting custom process environment.")
            #      self.process.setProcessEnvironment(process_environment)
            # else:
            #      # Use default environment if not set or UI hidden
            #      self.process.setProcessEnvironment(QProcessEnvironment()) # Ensure reset if previously set
            # --- End Optional Environment ---

            # --- Start Process ---
            # QProcess.start handles path finding and quoting for arguments list
            self.process.start(program, program_args)
            # Process state change signals (_handle_state_change, _handle_error) will indicate success/failure to start

        @pyqtSlot()
        def stop_script(self):
            """Stops the running process gracefully, then forcefully if necessary."""
            if self.process.state() == QProcess.ProcessState.Running:
                self.status_label.setText("Status: Attempting to stop...")
                logging.info("Attempting to terminate process...")
                self.process.terminate() # Ask nicely first

                # Set a short timer to check if termination worked, otherwise kill
                QTimer.singleShot(2000, self._force_kill_if_running) # Wait 2 seconds
            else:
                logging.debug("Stop clicked but process is not running.")

        def _force_kill_if_running(self):
            """Checks if the process is still running after terminate and kills it."""
            if self.process.state() == QProcess.ProcessState.Running:
                logging.warning("Process did not terminate after 2 seconds, killing forcefully.")
                self.process.kill()


        def _decode_output(self, byte_data: bytes) -> str:
            """Decodes byte data from process output using best-effort strategy."""
            try:
                # 1. Try UTF-8 first (most common)
                return byte_data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    # 2. Try system's preferred encoding
                    pref_enc = locale.getpreferredencoding(False)
                    if pref_enc and pref_enc.lower() != 'utf-8': # Avoid trying utf-8 again
                         logging.debug(f"Decoding output with system preferred encoding: {pref_enc}")
                         return byte_data.decode(pref_enc)
                    else:
                         raise # Re-raise if preferred is utf-8 or None
                except Exception:
                    try:
                        # 3. Fallback to UTF-8 with replacement characters
                        logging.warning("Could not decode output with UTF-8 or system encoding. Using replacement characters.")
                        return byte_data.decode('utf-8', errors='replace')
                    except Exception as e:
                        # 4. Absolute fallback: Representation
                        logging.error(f"Could not decode output bytes at all: {e}")
                        return repr(byte_data)

        @pyqtSlot()
        def _handle_stdout(self):
            """Append standard output to the text area, handling decoding."""
            if self.process:
                 byte_data = bytes(self.process.readAllStandardOutput())
                 text = self._decode_output(byte_data)
                 self.output_area.appendPlainText(text)
                 self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())

        @pyqtSlot()
        def _handle_stderr(self):
            """Append standard error to the text area, handling decoding."""
            if self.process:
                 byte_data = bytes(self.process.readAllStandardError())
                 text = self._decode_output(byte_data)
                 # Optionally, format stderr differently (e.g., color, prefix)
                 self.output_area.appendPlainText(f"[STDERR] {text}")
                 self.output_area.verticalScrollBar().setValue(self.output_area.verticalScrollBar().maximum())

        @pyqtSlot(QProcess.ProcessState)
        def _handle_state_change(self, state: QProcess.ProcessState):
            """Update UI elements based on process state changes."""
            logging.debug(f"Process state changed to: {state}")
            self._update_button_states(state) # Update button enables based on state
            # Update status label based on state
            if state == QProcess.ProcessState.NotRunning:
                # Let _handle_finished or _handle_error set the final status
                pass
            elif state == QProcess.ProcessState.Starting:
                self.status_label.setText("Status: Starting...")
            elif state == QProcess.ProcessState.Running:
                self.status_label.setText("Status: Running...")

        @pyqtSlot(int, QProcess.ExitStatus)
        def _handle_finished(self, exit_code: int, exit_status: QProcess.ExitStatus):
            """Handle process completion (normal exit or crash)."""
            status_text = "normally" if exit_status == QProcess.ExitStatus.NormalExit else "with crash"
            log_msg = f"Process finished {status_text}. Exit code: {exit_code}"
            logging.info(log_msg)
            self.output_area.appendPlainText(f"\n--- {log_msg} ---\n")
            self.status_label.setText(f"Status: Finished (Code: {exit_code})")
            # State should transition to NotRunning, which triggers button update via _handle_state_change


        @pyqtSlot(QProcess.ProcessError)
        def _handle_error(self, error: QProcess.ProcessError):
            """Handle errors reported by QProcess (e.g., failed to start, crashed)."""
            # These errors often occur *before* or *during* process start, or if it crashes.
            error_string = self.process.errorString() # Get descriptive error message
            logging.error(f"QProcess Error Occurred: {error} - {error_string}")
            self.output_area.appendPlainText(f"\n--- PROCESS ERROR: {error_string} ({error}) ---\n")
            self.status_label.setText(f"Status: Error ({error_string})")
            # State usually transitions to NotRunning after error, triggering button update

        def _update_button_states(self, state: QProcess.ProcessState):
            """Enable/disable buttons based on process state."""
            is_running_or_starting = (state == QProcess.ProcessState.Running or state == QProcess.ProcessState.Starting)
            self.run_button.setEnabled(not is_running_or_starting)
            self.stop_button.setEnabled(is_running_or_starting)
            # Disable input fields while running
            self.command_input.setEnabled(not is_running_or_starting)
            self.args_input.setEnabled(not is_running_or_starting)
            self.browse_button.setEnabled(not is_running_or_starting)
            # self.env_button.setEnabled(not is_running_or_starting) # Uncomment if using env vars UI

        def closeEvent(self, event):
            """Ensure the process is stopped and cleaned up when the dialog closes."""
            logging.debug("ExternalScriptRunnerDialog close event.")
            if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                logging.info("Dialog closing with process running. Attempting termination.")
                self.stop_script() # Ask nicely first
                # Wait a short time for process to finish after terminate/kill attempt
                if not self.process.waitForFinished(500): # Wait max 500ms
                    logging.warning("Process did not finish within 500ms of close event. May be orphaned.")
                    # Consider self.process.kill() here again if essential

            # Disconnect signals to prevent further handling after close
            if self.process:
                try: self.process.readyReadStandardOutput.disconnect(self._handle_stdout)
                except TypeError: pass
                try: self.process.readyReadStandardError.disconnect(self._handle_stderr)
                except TypeError: pass
                try: self.process.stateChanged.disconnect(self._handle_state_change)
                except TypeError: pass
                try: self.process.finished.disconnect(self._handle_finished)
                except TypeError: pass
                try: self.process.errorOccurred.disconnect(self._handle_error)
                except TypeError: pass

                # Schedule the QProcess object for deletion later by the event loop
                self.process.deleteLater()
                self.process = None # Clear reference

            super().closeEvent(event) # Accept the close event

else:
    # If Qt not available, provide a dummy class that raises error on instantiation
    class ExternalScriptRunnerDialog:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Cannot create ExternalScriptRunnerDialog: PyQt6 is not installed or available.")


# --- Standalone Test ---
if __name__ == "__main__":
    # Basic logging setup for testing this module
    log_level = logging.DEBUG # Use DEBUG for testing
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)-7s - [%(name)s:%(lineno)d] - %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().name = 'ExternalScriptRunnerTest'

    logging.info("--- Testing External Script Runner Dialog ---")

    # Test Dialog (only if Qt is available)
    if _QT_AVAILABLE:
        app = QApplication(sys.argv)
        dialog = ExternalScriptRunnerDialog()

        # --- Example Commands to Test ---
        if os.name == 'nt': # Windows examples
             # dialog.set_command("ping")
             # dialog.set_arguments("localhost -n 3")
             dialog.set_command("cmd")
             dialog.set_arguments("/c \"echo Hello from CMD & timeout /t 3 /nobreak > NUL & echo CMD Finished\"") # Command with spaces/quotes
             # dialog.set_command("C:\\Windows\\System32\\notepad.exe") # Path with spaces
             # dialog.set_arguments("\"C:\\Users\\Public\\Documents\\test file with spaces.txt\"") # Argument with spaces
        else: # Linux/macOS examples
             # dialog.set_command("ping")
             # dialog.set_arguments("localhost -c 3")
             dialog.set_command("bash")
             dialog.set_arguments("-c 'echo Hello from Bash; sleep 3; echo Bash Finished'")
             # dialog.set_command("ls")
             # dialog.set_arguments("-lha \"/tmp\"") # Example with quoted path arg

        # Example Python script (create dummy script first if needed)
        # test_py_path = Path("./dummy_test_script.py")
        # if not test_py_path.exists():
        #     with open(test_py_path, "w") as f:
        #         f.write('import sys, time\n')
        #         f.write('print(f"Hello from Python Script! Args: {sys.argv[1:]}")\n')
        #         f.write('print("Sleeping...", file=sys.stderr)\n')
        #         f.write('time.sleep(4)\n')
        #         f.write('print("Python Script finished.")\n')
        # dialog.set_command(str(test_py_path.resolve()))
        # dialog.set_arguments("'Argument with space' second_arg 123") # Test shlex parsing

        dialog.show()
        sys.exit(app.exec())
    else:
        print("\nPyQt6 not available, skipping ExternalScriptRunnerDialog test.")
        # Test instantiating the dummy class
        try:
            ExternalScriptRunnerDialog()
        except RuntimeError as e:
            print(f"Correctly caught expected error on instantiation: {e}")