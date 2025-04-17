import sys
import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QListWidget, QLabel,
                             QGridLayout, QTabWidget, QLineEdit, QPushButton, QComboBox,
                             QScrollArea, QWidget, QFileDialog, QRadioButton, QGroupBox,
                             QMenuBar, QMenu, QMessageBox, QDialog, QFormLayout, QDoubleSpinBox,
                             QCheckBox, QHBoxLayout)
# Suggestion 1: Use Qt6 backend for Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from airfoil_database.classes.AirfoilDatabase import AirfoilDatabase
from airfoil_database.classes.AirfoilSeries import AirfoilSeries

# Suggestion (Minor): Import specific function instead of wildcard if possible
# Assuming normalize_pointcloud is the main function needed here
from airfoil_database.xfoil.fix_airfoil_data import normalize_pointcloud # Or list other specific needed functions

class AirfoilViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.airfoil_db_instance = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Airfoil Viewer")
        self.setGeometry(100, 100, 1000, 800)

        self.layout = QVBoxLayout()

        self.setup_menu()
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.setup_start_tab()
        self.setup_viewer_tab()
        self.setup_geometry_search_tab()
        self.setup_compare_tab()
        self.setup_xfoil_tab()
        self.setup_xfoil_results_search_tab()
        # Suggestion 2: Removed call to setup_xfoil_plot()

        self.setLayout(self.layout)
        self.disable_other_tabs()

    def setup_menu(self):
        menubar = QMenuBar()
        file_menu = QMenu("File", self)
        menubar.addMenu(file_menu)

        open_action = file_menu.addAction("Open Database")
        open_action.triggered.connect(self.browse_database)

        save_action = file_menu.addAction("Save Database")
        save_action.triggered.connect(self.save_database)

        save_as_action = file_menu.addAction("Save Database As")
        save_as_action.triggered.connect(self.save_database_as)

        clear_db_action = file_menu.addAction("Clear Database")
        clear_db_action.triggered.connect(self.clear_database)

        self.layout.setMenuBar(menubar)

    def save_database(self):
        """Saves the current database."""
        if self.airfoil_db_instance:
            try:
                self.airfoil_db_instance.save_database()
                QMessageBox.information(self, "Database Saved", "Database saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error Saving Database", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Save Database", "No database is currently open.")

    def save_database_as(self):
        """Saves the current database to a new file."""
        if self.airfoil_db_instance:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Airfoil Database As", "", "SQLite Database (*.db)")
            if file_path:
                try:
                    self.airfoil_db_instance.save_database_as(file_path)
                    QMessageBox.information(self, "Database Saved As", "Database saved successfully.")
                except Exception as e:
                    QMessageBox.critical(self, "Error Saving Database", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Save Database As", "No database is currently open.")

    def clear_database(self):
        """Clears the current database."""
        if self.airfoil_db_instance:
            try:
                self.airfoil_db_instance.clear_database()
                QMessageBox.information(self, "Database Cleared", "Database cleared successfully.")
                self.disable_other_tabs()
                # Clear lists after clearing DB
                self.listWidget.clear()
                self.xfoil_airfoil_list.clear()
                # Consider clearing other lists (search results etc.) if necessary
            except Exception as e:
                QMessageBox.critical(self, "Error Clearing Database", f"An error occurred: {e}")
        else:
            QMessageBox.warning(self, "Clear Database", "No database is currently open.")

    # Suggestion 4: Merged load_airfoil_names into populate_airfoil_lists
    def populate_airfoil_lists(self):
        """Populates all airfoil lists in the UI and enables viewer widgets."""
        # Clear existing lists first
        self.listWidget.clear()
        if hasattr(self, 'xfoil_airfoil_list'): # Check if list exists before clearing
             self.xfoil_airfoil_list.clear()
        # Add clearing for other lists like search results if needed

        if self.airfoil_db_instance:
            try:
                with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
                    cursor = conn.cursor()
                    # Fetch names ordered for consistency
                    cursor.execute("SELECT name FROM airfoils ORDER BY name")
                    results = cursor.fetchall()
                    airfoil_names = [row[0] for row in results]

                    if hasattr(self, 'listWidget'):
                        self.listWidget.addItems(airfoil_names)
                    if hasattr(self, 'xfoil_airfoil_list'):
                         self.xfoil_airfoil_list.addItems(airfoil_names)
                    # Add items to other lists if they exist

                # Enable viewer widgets (logic from old load_airfoil_names)
                if hasattr(self, 'viewer_name_edit'): # Check if widgets exist
                    self.viewer_name_edit.setEnabled(True)
                    self.viewer_desc_edit.setEnabled(True)
                    self.viewer_series_combo.setEnabled(True)
                    self.viewer_source_edit.setEnabled(True)
                    self.update_info_button.setEnabled(True)

            except sqlite3.Error as e:
                QMessageBox.critical(self, "Database Error", f"Error populating airfoil lists: {e}")
            except AttributeError as e:
                 QMessageBox.critical(self, "Attribute Error", f"Error accessing UI element during list population: {e}")
        else:
             # Disable viewer widgets if no DB instance
             if hasattr(self, 'viewer_name_edit'):
                 self.viewer_name_edit.setEnabled(False)
                 self.viewer_desc_edit.setEnabled(False)
                 self.viewer_series_combo.setEnabled(False)
                 self.viewer_source_edit.setEnabled(False)
                 self.update_info_button.setEnabled(False)


    def setup_start_tab(self):
        start_tab = QWidget()
        start_layout = QVBoxLayout()

        # Database Selection
        db_group = QGroupBox("Database Selection")
        db_layout = QVBoxLayout()

        self.db_path_edit = QLineEdit()
        db_layout.addWidget(self.db_path_edit)

        db_browse_button = QPushButton("Browse")
        db_browse_button.clicked.connect(self.browse_database)
        db_layout.addWidget(db_browse_button)

        db_group.setLayout(db_layout)
        start_layout.addWidget(db_group)

        # File Import
        file_group = QGroupBox("Import Airfoils")
        file_layout = QVBoxLayout()

        self.file_path_edit = QLineEdit()
        file_layout.addWidget(self.file_path_edit)

        file_browse_button = QPushButton("Browse")
        file_browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(file_browse_button)

        self.overwrite_checkbox = QRadioButton("Overwrite Existing")
        file_layout.addWidget(self.overwrite_checkbox)

        import_button = QPushButton("Import")
        import_button.clicked.connect(self.import_file)
        file_layout.addWidget(import_button)

        file_group.setLayout(file_layout)
        start_layout.addWidget(file_group)

        # Individual Airfoil Addition
        airfoil_group = QGroupBox("Add Individual Airfoil")
        airfoil_layout = QGridLayout()

        airfoil_layout.addWidget(QLabel("Name:"), 0, 0)
        self.airfoil_name_edit = QLineEdit()
        airfoil_layout.addWidget(self.airfoil_name_edit, 0, 1)

        airfoil_layout.addWidget(QLabel("Description:"), 1, 0)
        self.airfoil_desc_edit = QLineEdit()
        airfoil_layout.addWidget(self.airfoil_desc_edit, 1, 1)

        airfoil_layout.addWidget(QLabel("Airfoil Series:"), 2, 0)
        self.airfoil_series_combo = QComboBox()
        for series in AirfoilSeries:
            self.airfoil_series_combo.addItem(series.value)
        airfoil_layout.addWidget(self.airfoil_series_combo, 2, 1)

        airfoil_layout.addWidget(QLabel("Source:"), 3, 0)
        self.airfoil_source_edit = QLineEdit()
        airfoil_layout.addWidget(self.airfoil_source_edit, 3, 1)

        airfoil_layout.addWidget(QLabel("Point Cloud File:"), 4, 0)
        self.airfoil_pointcloud_edit = QLineEdit()
        airfoil_layout.addWidget(self.airfoil_pointcloud_edit, 4, 1)

        pointcloud_browse_button = QPushButton("Browse")
        pointcloud_browse_button.clicked.connect(self.browse_pointcloud)
        airfoil_layout.addWidget(pointcloud_browse_button, 4, 2)

        add_airfoil_button = QPushButton("Add Airfoil")
        add_airfoil_button.clicked.connect(self.add_individual_airfoil)
        airfoil_layout.addWidget(add_airfoil_button, 5, 0, 1, 3)

        airfoil_group.setLayout(airfoil_layout)
        start_layout.addWidget(airfoil_group)

        # Load Database Button (from path edit)
        load_button = QPushButton("Load Database from Path")
        load_button.clicked.connect(self.load_database_from_path) # Renamed function for clarity
        start_layout.addWidget(load_button)

        start_tab.setLayout(start_layout)
        self.tabs.addTab(start_tab, "Start")

    def browse_database(self):
        """Opens a file dialog to select and load a database."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Airfoil Database", "", "SQLite Database (*.db)")
        if file_path:
            try:
                self.airfoil_db_instance = AirfoilDatabase(file_path) # Use full path
                self.db_path_edit.setText(file_path) # Update text field
                self.enable_other_tabs()
                self.populate_airfoil_lists() # Use the consolidated method
                QMessageBox.information(self, "Database Opened", "Database opened successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error Opening Database", f"An error occurred: {e}")
                self.airfoil_db_instance = None # Ensure instance is None on error
                self.disable_other_tabs()
                self.populate_airfoil_lists() # Update lists (will be empty)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "CSV Files (*.csv);;JSON Files (*.json);;All Files (*.*)")
        if file_path:
            self.file_path_edit.setText(file_path)

    def import_file(self):
        if not self.airfoil_db_instance:
             QMessageBox.warning(self, "Import File", "Please open or load a database first.")
             return

        file_path = self.file_path_edit.text()
        if not file_path:
            QMessageBox.warning(self, "Import File", "Please select a file to import.")
            return

        overwrite = self.overwrite_checkbox.isChecked()
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        try:
            if file_extension == '.csv':
                self.airfoil_db_instance.add_airfoils_from_csv(file_path, overwrite)
            elif file_extension == '.json':
                self.airfoil_db_instance.add_airfoils_from_json(file_path, overwrite)
            else:
                QMessageBox.warning(self, "Import File", "Unsupported file type. Please select a CSV or JSON file.")
                return
            QMessageBox.information(self, "Import Complete", f"Imported data from {os.path.basename(file_path)}.")
            self.populate_airfoil_lists() # Refresh lists after import
        except Exception as e:
             QMessageBox.critical(self, "Import Error", f"An error occurred during import: {e}")

    # Renamed function for clarity
    def load_database_from_path(self):
        """Loads a database from the path specified in the QLineEdit."""
        db_path = self.db_path_edit.text()
        if not db_path:
            QMessageBox.warning(self, "Load Database", "Please enter or browse for a database path.")
            return
        if not os.path.exists(db_path):
            QMessageBox.warning(self, "Load Database", f"Database file not found at: {db_path}")
            return

        try:
            # Suggestion 4: Consistency - Use full path like browse_database
            self.airfoil_db_instance = AirfoilDatabase(db_name=db_path)
            self.enable_other_tabs()
            self.populate_airfoil_lists() # Use the consolidated method
            QMessageBox.information(self, "Database Loaded", "Database loaded successfully from path.")
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Database", f"An error occurred: {e}")
            self.airfoil_db_instance = None # Ensure instance is None on error
            self.disable_other_tabs()
            self.populate_airfoil_lists() # Update lists (will be empty)

    # Suggestion 4: Removed redundant load_airfoil_names method

    def browse_pointcloud(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Point Cloud File", "", "Text Files (*.txt);;All Files (*.*)")
        if file_path:
            self.airfoil_pointcloud_edit.setText(file_path)

    def add_individual_airfoil(self):
        if not self.airfoil_db_instance:
            QMessageBox.warning(self, "Add Airfoil", "Please load a database first.")
            return

        name = self.airfoil_name_edit.text().strip()
        description = self.airfoil_desc_edit.text().strip()
        series = self.airfoil_series_combo.currentText() #Get the value from the combo box.
        source = self.airfoil_source_edit.text().strip()
        pointcloud_file = self.airfoil_pointcloud_edit.text().strip()

        if not name:
            QMessageBox.warning(self, "Add Airfoil", "Airfoil Name is required.")
            return
        if not pointcloud_file:
            QMessageBox.warning(self, "Add Airfoil", "Point Cloud File is required.")
            return
        if not os.path.exists(pointcloud_file):
             QMessageBox.warning(self, "Add Airfoil", f"Point Cloud File not found: {pointcloud_file}")
             return

        try:
            # It's generally safer to read point cloud data within the database method
            # Pass the file path instead of reading here, let AirfoilDatabase handle it
            self.airfoil_db_instance.store_airfoil_data(name, description, pointcloud_file, series, source, is_file_path=True)
            self.populate_airfoil_lists() # Use the consolidated method
            QMessageBox.information(self, "Add Airfoil", f"Airfoil '{name}' added successfully.")
            # Clear input fields after successful addition
            self.airfoil_name_edit.clear()
            self.airfoil_desc_edit.clear()
            self.airfoil_source_edit.clear()
            self.airfoil_pointcloud_edit.clear()
            self.airfoil_series_combo.setCurrentIndex(0)

        except FileNotFoundError: # Should be caught by os.path.exists check now
            QMessageBox.warning(self, "Add Airfoil", f"Point Cloud File not found: {pointcloud_file}")
        except sqlite3.IntegrityError:
             QMessageBox.warning(self, "Add Airfoil", f"Airfoil name '{name}' already exists. Choose a different name or enable overwrite if applicable.")
        except Exception as e:
             QMessageBox.critical(self, "Add Airfoil Error", f"An error occurred: {e}")


    def setup_viewer_tab(self):
        viewer_tab = QWidget()
        viewer_layout = QGridLayout()

        self.listWidget = QListWidget()
        self.listWidget.itemClicked.connect(self.plot_airfoil)
        viewer_layout.addWidget(self.listWidget, 0, 0, 2, 1)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        # Add toolbar for plot navigation
        self.viewer_toolbar = NavigationToolbar(self.canvas, self)
        viewer_layout.addWidget(self.viewer_toolbar, 0, 1, 1, 3) # Place toolbar above canvas
        viewer_layout.addWidget(self.canvas, 1, 1, 1, 3) # Place canvas below toolbar


        # Airfoil Geometry and Info
        info_group = QGroupBox("Airfoil Geometry and Info")
        info_layout = QGridLayout()

        # Geometry Labels
        self.geometry_labels = {
            "Max Thickness:": QLabel("N/A"),
            "Max Camber:": QLabel("N/A"),
            "Chord Length:": QLabel("N/A"),
            "Span:": QLabel("N/A"),
            "Aspect Ratio:": QLabel("N/A"),
            "Leading Edge Radius:": QLabel("N/A"),
            "Trailing Edge Angle:": QLabel("N/A"),
            "Thickness/Chord Ratio:": QLabel("N/A")
        }

        row = 0
        col = 0
        for label_text, label_widget in self.geometry_labels.items():
            info_layout.addWidget(QLabel(label_text), row, col)
            info_layout.addWidget(label_widget, row, col + 1)
            col += 2 # Place label and value side-by-side
            if col >= 4 : # Move to next row after two pairs
                row += 1
                col = 0

        # Airfoil Info Edits (Place below geometry)
        info_layout.addWidget(QLabel("Name:"), row, 0)
        self.viewer_name_edit = QLineEdit()
        self.viewer_name_edit.setEnabled(False) # Initially disabled
        info_layout.addWidget(self.viewer_name_edit, row, 1)

        row += 1
        info_layout.addWidget(QLabel("Description:"), row, 0)
        self.viewer_desc_edit = QLineEdit()
        self.viewer_desc_edit.setEnabled(False) # Initially disabled
        info_layout.addWidget(self.viewer_desc_edit, row, 1)

        row += 1
        info_layout.addWidget(QLabel("Airfoil Series:"), row, 0)
        self.viewer_series_combo = QComboBox()
        for series in AirfoilSeries:
            self.viewer_series_combo.addItem(series.value)
        self.viewer_series_combo.setEnabled(False) # Initially disabled
        info_layout.addWidget(self.viewer_series_combo, row, 1)

        row += 1
        info_layout.addWidget(QLabel("Source:"), row, 0)
        self.viewer_source_edit = QLineEdit()
        self.viewer_source_edit.setEnabled(False) # Initially disabled
        info_layout.addWidget(self.viewer_source_edit, row, 1)

        row += 1
        self.update_info_button = QPushButton("Update Info")
        self.update_info_button.clicked.connect(self.update_airfoil_info)
        self.update_info_button.setEnabled(False) # Initially disabled
        info_layout.addWidget(self.update_info_button, row, 0, 1, 2) # Span across 2 columns

        info_group.setLayout(info_layout)
        # Place info group below the plot
        viewer_layout.addWidget(info_group, 2, 1, 1, 3) # Row 2, Col 1, Span 1 row, 3 cols

        # Adjust row/column spans for list widget and plot area
        viewer_layout.setRowStretch(1, 4) # Give more stretch to plot row
        viewer_layout.setRowStretch(2, 1) # Less stretch to info group row
        viewer_layout.setColumnStretch(0, 1) # List widget width
        viewer_layout.setColumnStretch(1, 4) # Plot area width

        viewer_tab.setLayout(viewer_layout)
        self.tabs.addTab(viewer_tab, "Viewer")

    def update_airfoil_info(self):
        selected_item = self.listWidget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "Update Info", "Please select an airfoil from the list.")
            return

        original_name = selected_item.text()
        new_name = self.viewer_name_edit.text().strip()
        description = self.viewer_desc_edit.text().strip()
        series = self.viewer_series_combo.currentText()
        source = self.viewer_source_edit.text().strip()

        if not new_name:
             QMessageBox.warning(self, "Update Info", "Airfoil name cannot be empty.")
             return

        try:
            self.airfoil_db_instance.update_airfoil_info(original_name, new_name, description, series, source)
            QMessageBox.information(self, "Update Info", f"Airfoil '{original_name}' info updated.")
            # Refresh lists to reflect potential name change
            current_index = self.listWidget.row(selected_item) # Store index
            self.populate_airfoil_lists() # This reloads all lists
            # Re-select the item (might be at a different index if name changed/sorted)
            items = self.listWidget.findItems(new_name, Qt.MatchFlag.MatchExactly)
            if items:
                self.listWidget.setCurrentItem(items[0])
            else: # If not found (shouldn't happen unless error), select previous index
                 if current_index < self.listWidget.count():
                      self.listWidget.setCurrentRow(current_index)

        except sqlite3.IntegrityError:
             QMessageBox.warning(self, "Update Info", f"Airfoil name '{new_name}' already exists. Choose a different name.")
        except Exception as e:
             QMessageBox.critical(self, "Update Error", f"An error occurred while updating info: {e}")


    def plot_airfoil(self, item):
        if not self.airfoil_db_instance: return # Should not happen if list is populated

        airfoil_name = item.text()
        self.ax.clear()
        try:
            self.airfoil_db_instance.plot_airfoil(airfoil_name, self.ax)
            # Suggestion 4: Removed call to display_geometry_data
            self.populate_info_edits(airfoil_name) # This now handles geometry display too
            self.ax.set_xlabel("X Coordinate")
            self.ax.set_ylabel("Y Coordinate")
            self.ax.set_title(f"Airfoil: {airfoil_name}")
            self.ax.grid(True)
            self.ax.axis('equal')
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"Could not plot airfoil '{airfoil_name}': {e}")
            self.ax.clear() # Clear plot on error
            self.ax.set_title("Plotting Error")
            self.canvas.draw()
            self.populate_info_edits(None) # Clear info fields


    def populate_info_edits(self, airfoil_name):
        """Populates the info edit fields AND geometry labels for the selected airfoil."""
        if airfoil_name and self.airfoil_db_instance:
            try:
                # Fetch basic info
                data = self.airfoil_db_instance.get_airfoil_data(airfoil_name)
                if data and len(data) >= 4: # Check length >= 4 for safety
                    description, _, series, source = data[:4] # Get first 4 elements
                    self.viewer_name_edit.setText(airfoil_name)
                    self.viewer_desc_edit.setText(description if description else "")
                    self.viewer_series_combo.setCurrentText(series if series else "")
                    self.viewer_source_edit.setText(source if source else "")
                else:
                    # Clear fields if basic data not found
                    self.viewer_name_edit.setText(airfoil_name) # Keep name
                    self.viewer_desc_edit.clear()
                    self.viewer_series_combo.setCurrentIndex(0)
                    self.viewer_source_edit.clear()

                # Fetch and Populate Geometry Data
                with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT max_thickness, max_camber, chord_length, span, aspect_ratio,
                               leading_edge_radius, trailing_edge_angle, thickness_to_chord_ratio
                        FROM airfoil_geometry
                        WHERE name = ?
                    """, (airfoil_name,))
                    geometry_data = cursor.fetchone()

                if geometry_data:
                    labels = list(self.geometry_labels.values())
                    for i, value in enumerate(geometry_data):
                        if value is not None:
                            try:
                                # Format to reasonable precision
                                labels[i].setText("{:.4f}".format(value))
                            except (ValueError, TypeError):
                                labels[i].setText("Invalid Data") # Handle non-numeric data
                        else:
                             labels[i].setText("N/A") # Use N/A for None
                else:
                    # Set all geometry labels to N/A if no geometry data row
                    for label in self.geometry_labels.values():
                        label.setText("N/A")

            except sqlite3.Error as e:
                 QMessageBox.critical(self, "Database Error", f"Error fetching data for '{airfoil_name}': {e}")
                 self._clear_viewer_fields() # Clear fields on DB error
            except Exception as e:
                 QMessageBox.critical(self, "Error", f"An unexpected error occurred populating info: {e}")
                 self._clear_viewer_fields()

        else:
            # Clear all fields if no airfoil_name or no db instance
            self._clear_viewer_fields()

    def _clear_viewer_fields(self):
        """Helper function to clear all viewer fields."""
        self.viewer_name_edit.clear()
        self.viewer_desc_edit.clear()
        self.viewer_series_combo.setCurrentIndex(-1) # No selection
        self.viewer_source_edit.clear()
        for label in self.geometry_labels.values():
            label.setText("N/A")


    def setup_geometry_search_tab(self):
        search_tab = QWidget()
        search_layout = QGridLayout()

        self.search_params = {}
        self.search_fields = ["max_thickness", "max_camber", "chord_length", "span",
                               "aspect_ratio", "leading_edge_radius", "trailing_edge_angle",
                               "thickness_to_chord_ratio"]

        row = 0
        # Add input fields for geometry parameters
        for field in self.search_fields:
            search_layout.addWidget(QLabel(field.replace("_", " ").title() + ":"), row, 0)
            self.search_params[field] = QLineEdit()
            search_layout.addWidget(self.search_params[field], row, 1)
            row += 1

        # Suggestion (from previous review): Add Tolerance Inputs
        search_layout.addWidget(QLabel("Tolerance:"), row, 0)
        self.geometry_tolerance_edit = QLineEdit("0.1") # Default tolerance
        search_layout.addWidget(self.geometry_tolerance_edit, row, 1)
        row += 1

        search_layout.addWidget(QLabel("Tolerance Type:"), row, 0)
        self.geometry_tolerance_type_combo = QComboBox()
        self.geometry_tolerance_type_combo.addItems(["absolute", "percentage"])
        search_layout.addWidget(self.geometry_tolerance_type_combo, row, 1)
        row += 1


        # Buttons
        self.search_button = QPushButton("Search by Geometry")
        self.search_button.clicked.connect(self.search_airfoils)
        search_layout.addWidget(self.search_button, row, 0, 1, 2)
        row += 1

        self.clear_criteria_button = QPushButton("Clear Criteria")
        self.clear_criteria_button.clicked.connect(self.clear_search_criteria)
        search_layout.addWidget(self.clear_criteria_button, row, 0, 1, 2)
        row += 1

        # Make input section take less vertical space
        search_layout.setRowStretch(row, 1) # Add stretch after inputs

        # Results List (Column 2)
        search_layout.addWidget(QLabel("Search Results:"), 0, 2) # Add label
        self.search_results_list = QListWidget()
        self.search_results_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection) # Allow multi-select
        search_layout.addWidget(self.search_results_list, 1, 2, row-1, 1) # Span rows, col 2

        # List Clear Button (Below List)
        self.clear_list_button = QPushButton("Clear List")
        self.clear_list_button.clicked.connect(self.clear_search_list)
        search_layout.addWidget(self.clear_list_button, row, 2) # Same row as stretch

        # Add Matplotlib figure and canvas (Column 3 & 4)
        self.search_fig, self.search_ax = plt.subplots()
        self.search_canvas = FigureCanvas(self.search_fig)
        self.search_toolbar = NavigationToolbar(self.search_canvas, self)
        search_layout.addWidget(self.search_toolbar, 0, 3, 1, 2) # Toolbar above
        search_layout.addWidget(self.search_canvas, 1, 3, row-1, 2) # Canvas below, span cols 3,4

        # Plot Buttons (Below Plot)
        self.clear_plot_button = QPushButton("Clear Plot")
        self.clear_plot_button.clicked.connect(self.clear_search_plot)
        search_layout.addWidget(self.clear_plot_button, row, 3)

        self.plot_selected_button = QPushButton("Plot Selected")
        self.plot_selected_button.clicked.connect(self.plot_selected_airfoils_search_tab)
        search_layout.addWidget(self.plot_selected_button, row, 4)

        # Adjust column stretches for responsiveness
        search_layout.setColumnStretch(1, 1) # Input value column
        search_layout.setColumnStretch(2, 2) # Results list column
        search_layout.setColumnStretch(3, 3) # Plot area columns
        search_layout.setColumnStretch(4, 3) # Plot area columns

        search_tab.setLayout(search_layout)
        self.tabs.addTab(search_tab, "Geometry Search")

    # Suggestion 4: Removed redundant display_geometry_data method

    def search_airfoils(self):
        if not self.airfoil_db_instance:
            QMessageBox.warning(self, "Search Error", "Please open a database first.")
            return

        search_criteria = {}
        # Suggestion 7: Validate tolerance inputs first
        try:
            tolerance = float(self.geometry_tolerance_edit.text())
            if tolerance < 0:
                 raise ValueError("Tolerance cannot be negative")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid tolerance value: {e}. Please enter a non-negative number.")
            return
        tolerance_type = self.geometry_tolerance_type_combo.currentText()


        # Collect search parameters and validate numeric inputs
        has_criteria = False
        for field, line_edit in self.search_params.items():
            value_str = line_edit.text().strip()
            if value_str:
                try:
                    search_criteria[field] = float(value_str)
                    has_criteria = True
                except ValueError:
                    # Suggestion 7: Add input validation
                    QMessageBox.warning(self, "Input Error", f"Invalid numeric value entered for '{field.replace('_', ' ').title()}'. Please correct it.")
                    return # Stop search if any input is invalid

        if not has_criteria:
             QMessageBox.information(self, "Search", "Please enter at least one geometry parameter value to search for.")
             return

        # Clear previous results
        self.search_results_list.clear()
        self.clear_search_plot() # Also clear plot on new search

        # --- Search Logic ---
        # Decide between AND / OR logic. Current implementation below is OR.
        # For OR logic (find airfoils matching ANY criteria):
        all_results = set()
        try:
            for field, target_value in search_criteria.items():
                # Use the user-provided tolerance values
                airfoil_names = self.airfoil_db_instance.find_airfoils_by_geometry(
                    field, target_value, tolerance, tolerance_type
                )
                if airfoil_names: # Only update if results are found for this criterion
                    all_results.update(airfoil_names)

            # Populate the list widget with the combined results
            if all_results:
                for name in sorted(list(all_results)):
                    self.search_results_list.addItem(name)
                QMessageBox.information(self, "Search Complete", f"Found {len(all_results)} airfoil(s) matching one or more criteria.")
            else:
                QMessageBox.information(self, "Search Complete", "No airfoils found matching the specified criteria.")

        except Exception as e:
            QMessageBox.critical(self, "Search Error", f"An error occurred during search: {e}")


        # # For AND logic (find airfoils matching ALL criteria - more complex):
        # # This would require fetching all geometry data and filtering, or complex SQL
        # # Example (Conceptual - needs backend support or more complex logic):
        # try:
        #     matching_airfoils = self.airfoil_db_instance.find_airfoils_matching_all_geometry(
        #         search_criteria, tolerance, tolerance_type
        #     ) # Assumes such a method exists in AirfoilDatabase
        #     if matching_airfoils:
        #         self.search_results_list.addItems(sorted(matching_airfoils))
        #         QMessageBox.information(self, "Search Complete", f"Found {len(matching_airfoils)} airfoil(s) matching all criteria.")
        #     else:
        #         QMessageBox.information(self, "Search Complete", "No airfoils found matching all specified criteria.")
        # except Exception as e:
        #     QMessageBox.critical(self, "Search Error", f"An error occurred during search: {e}")


    def plot_selected_airfoils_search_tab(self):
        if not self.airfoil_db_instance: return
        selected_items = self.search_results_list.selectedItems()
        names = [item.text() for item in selected_items]
        if names:
            self.search_ax.clear()
            plotted_count = 0
            max_plots = 10 # Limit number of airfoils to plot at once for clarity
            if len(names) > max_plots:
                 QMessageBox.information(self, "Plot Limit", f"Plotting the first {max_plots} selected airfoils.")
                 names = names[:max_plots]

            for name in names:
                try:
                    # Use a consistent style or vary it if needed
                    self.airfoil_db_instance.add_airfoil_to_plot(name, self.search_ax, linestyle='-', marker=None, markersize=3)
                    plotted_count += 1
                except Exception as e:
                    print(f"Warning: Could not plot airfoil '{name}' from search results: {e}") # Log warning

            if plotted_count > 0:
                self.search_ax.set_xlabel("X Coordinate")
                self.search_ax.set_ylabel("Y Coordinate")
                self.search_ax.set_title("Selected Airfoil Comparison")
                self.search_ax.grid(True)
                self.search_ax.axis('equal')
                self.search_ax.legend()
                self.search_canvas.draw()
            else:
                 # Clear if no airfoils could be plotted
                 self.clear_search_plot()
                 QMessageBox.warning(self, "Plot Error", "Could not plot any of the selected airfoils.")

        else:
            QMessageBox.information(self, "Plot Selected", "Please select one or more airfoils from the results list to plot.")


    def clear_search_criteria(self):
        for line_edit in self.search_params.values():
            line_edit.clear()
        # Also clear tolerance fields if added
        if hasattr(self, 'geometry_tolerance_edit'):
             self.geometry_tolerance_edit.setText("0.1") # Reset default
             self.geometry_tolerance_type_combo.setCurrentIndex(0) # Reset default


    def clear_search_list(self):
        self.search_results_list.clear()

    def clear_search_plot(self):
        self.search_ax.clear()
        self.search_ax.set_title("Airfoil Plot Area") # Reset title
        self.search_ax.set_xlabel("X Coordinate")
        self.search_ax.set_ylabel("Y Coordinate")
        self.search_ax.grid(False) # Turn off grid when clear
        self.search_canvas.draw()

    def disable_other_tabs(self):
        """Disables other tabs until a database is opened."""
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)

    def enable_other_tabs(self):
        """Enables other tabs after a database is opened."""
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, True)

    def setup_compare_tab(self):
        compare_tab = QWidget()
        compare_layout = QVBoxLayout()

        # Point Cloud Selection
        pointcloud_group = QGroupBox("Point Cloud Selection")
        pointcloud_layout = QGridLayout()

        pointcloud_layout.addWidget(QLabel("Point Cloud File:"), 0, 0)
        self.compare_pointcloud_edit = QLineEdit()
        self.compare_pointcloud_edit.setPlaceholderText("Browse for a point cloud file (.txt)...")
        pointcloud_layout.addWidget(self.compare_pointcloud_edit, 0, 1)

        pointcloud_browse_button = QPushButton("Browse")
        pointcloud_browse_button.clicked.connect(self.browse_compare_pointcloud)
        pointcloud_layout.addWidget(pointcloud_browse_button, 0, 2)

        pointcloud_group.setLayout(pointcloud_layout)
        compare_layout.addWidget(pointcloud_group)

        # Plotting Area
        self.compare_fig, self.compare_ax = plt.subplots()
        self.compare_canvas = FigureCanvas(self.compare_fig)
        self.compare_toolbar = NavigationToolbar(self.compare_canvas, self)
        compare_layout.addWidget(self.compare_toolbar)
        compare_layout.addWidget(self.compare_canvas)

        # Buttons below plot
        button_layout = QHBoxLayout() # Horizontal layout for buttons
        self.compare_button = QPushButton("Find Best Match in DB")
        self.compare_button.clicked.connect(self.compare_airfoils)
        self.compare_button.setEnabled(False) # Disable until point cloud loaded

        self.clear_compare_plot_button = QPushButton("Clear Plot")
        self.clear_compare_plot_button.clicked.connect(self.clear_compare_plot)

        button_layout.addWidget(self.compare_button)
        button_layout.addWidget(self.clear_compare_plot_button)
        button_layout.addStretch() # Add stretch to push buttons left
        compare_layout.addLayout(button_layout)

        compare_tab.setLayout(compare_layout)
        self.tabs.addTab(compare_tab, "Compare")

        self.compare_canvas.mpl_connect('button_press_event', self.on_compare_click)
        self.compare_pointcloud_points = None # Store the numpy array
        self.clear_compare_plot() # Initialize plot


    def browse_compare_pointcloud(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Point Cloud File", "", "Text Files (*.txt);;DAT Files (*.dat);;All Files (*.*)")
        if file_path:
            self.compare_pointcloud_edit.setText(file_path)
            self.load_compare_pointcloud(file_path)

    def load_compare_pointcloud(self, file_path):
        try:
            # Assuming normalize_pointcloud handles reading and processing
            # It might be better if AirfoilDatabase or a utility class handles this
            points_raw = np.loadtxt(file_path, skiprows=1) # Skip header line if typical .dat format
            normalized_points = normalize_pointcloud(points_raw) # Use imported function

            self.compare_pointcloud_points = normalized_points

            self.compare_ax.clear()
            self.compare_ax.plot(normalized_points[:, 0], normalized_points[:, 1], 'bo-', markersize=3, label="Input Point Cloud (Normalized)")
            self.compare_ax.set_xlabel("X Coordinate")
            self.compare_ax.set_ylabel("Y Coordinate")
            self.compare_ax.set_title("Loaded Point Cloud")
            self.compare_ax.grid(True)
            self.compare_ax.axis('equal')
            self.compare_ax.legend()
            self.compare_canvas.draw()
            self.compare_button.setEnabled(True) # Enable compare button

        except FileNotFoundError:
            QMessageBox.warning(self, "Load Point Cloud", f"File not found: {file_path}")
            self.compare_pointcloud_points = None
            self.compare_button.setEnabled(False)
            self.clear_compare_plot()
        except Exception as e:
            QMessageBox.critical(self, "Load Point Cloud Error", f"Error loading or processing point cloud: {e}")
            self.compare_pointcloud_points = None
            self.compare_button.setEnabled(False)
            self.clear_compare_plot()

    def compare_airfoils(self):
        if self.compare_pointcloud_points is None:
            QMessageBox.warning(self, "Compare Airfoils", "Please load a point cloud first.")
            return
        if not self.airfoil_db_instance:
             QMessageBox.warning(self, "Compare Airfoils", "Please open a database first.")
             return

        try:
            # Convert numpy array back to string format expected by find_best_matching_airfoils
            # Or modify find_best_matching_airfoils to accept a numpy array directly (preferred)
            pointcloud_str = '\n'.join([' '.join(map(str, point)) for point in self.compare_pointcloud_points])

            matches = self.airfoil_db_instance.find_best_matching_airfoils(pointcloud_str, top_n=1) # Find top 1 match

            if matches:
                best_match_name, score = matches[0]
                # Re-plot the input cloud and add the best match
                self.compare_ax.clear()
                self.compare_ax.plot(self.compare_pointcloud_points[:, 0], self.compare_pointcloud_points[:, 1], 'bo-', markersize=3, label="Input Point Cloud")
                # Plot the best match from DB
                self.airfoil_db_instance.add_airfoil_to_plot(best_match_name, self.compare_ax, linestyle='r--', label=f"Best Match: {best_match_name} (Score: {score:.4f})")

                self.compare_ax.set_xlabel("X Coordinate")
                self.compare_ax.set_ylabel("Y Coordinate")
                self.compare_ax.set_title("Point Cloud Comparison")
                self.compare_ax.grid(True)
                self.compare_ax.axis('equal')
                self.compare_ax.legend()
                self.compare_canvas.draw()
                QMessageBox.information(self, "Compare Complete", f"Best matching airfoil found: {best_match_name} (Similarity Score: {score:.4f})")
            else:
                QMessageBox.information(self, "Compare Airfoils", "No matching airfoils found in the database.")
                # Keep the input plot visible
                self.compare_ax.clear()
                self.compare_ax.plot(self.compare_pointcloud_points[:, 0], self.compare_pointcloud_points[:, 1], 'bo-', markersize=3, label="Input Point Cloud")
                self.compare_ax.set_title("Input Point Cloud (No Match Found)")
                self.compare_ax.legend()
                self.compare_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", f"An error occurred during comparison: {e}")

    def clear_compare_plot(self):
         self.compare_ax.clear()
         self.compare_ax.set_title("Load Point Cloud to Compare")
         self.compare_ax.set_xlabel("X Coordinate")
         self.compare_ax.set_ylabel("Y Coordinate")
         self.compare_ax.grid(True)
         self.compare_ax.axis('equal')
         self.compare_canvas.draw()
         self.compare_pointcloud_points = None # Clear loaded points
         self.compare_button.setEnabled(False) # Disable compare button
         self.compare_pointcloud_edit.clear() # Clear file path


    def on_compare_click(self, event):
        """Handles clicks on the compare plot for point editing."""
        if event.inaxes != self.compare_ax or event.button != 1 or self.compare_pointcloud_points is None:
            return # Ignore clicks outside axes, not left button, or if no points loaded

        x, y = event.xdata, event.ydata
        if x is None or y is None: return # Ignore clicks outside plot area

        # Find the index of the closest point in the loaded point cloud
        distances = np.linalg.norm(self.compare_pointcloud_points - np.array([x, y]), axis=1)
        closest_point_index = np.argmin(distances)
        # Define a threshold distance to consider it a click "on" a point
        click_threshold = (self.compare_ax.get_xlim()[1] - self.compare_ax.get_xlim()[0]) / 50 # Heuristic threshold
        if distances[closest_point_index] > click_threshold:
             return # Click was too far from any point

        # Get current coordinates of the closest point
        current_x = self.compare_pointcloud_points[closest_point_index, 0]
        current_y = self.compare_pointcloud_points[closest_point_index, 1]

        # Open the edit dialog
        dialog = PointEditDialog(current_x, current_y, self)
        if dialog.exec(): # User clicked OK
            new_x, new_y = dialog.get_values()
            # Update the point coordinates
            self.compare_pointcloud_points[closest_point_index, 0] = new_x
            self.compare_pointcloud_points[closest_point_index, 1] = new_y

            # Re-plot the modified point cloud (clears previous plots like best match)
            self.compare_ax.clear()
            self.compare_ax.plot(self.compare_pointcloud_points[:, 0], self.compare_pointcloud_points[:, 1], 'bo-', markersize=3, label="Input Point Cloud (Edited)")
            self.compare_ax.set_xlabel("X Coordinate")
            self.compare_ax.set_ylabel("Y Coordinate")
            self.compare_ax.set_title("Point Cloud (Edited)")
            self.compare_ax.grid(True)
            self.compare_ax.axis('equal')
            self.compare_ax.legend()
            self.compare_canvas.draw()
            # Note: After editing, the "best match" plot is cleared. User needs to compare again.
            QMessageBox.information(self, "Point Edited", "Point coordinates updated. Re-run 'Find Best Match' if needed.")

    def setup_xfoil_tab(self):
        """Sets up the XFOIL Results tab."""
        xfoil_tab = QWidget()
        xfoil_layout = QGridLayout() # Use grid for better alignment

        # Airfoil Selection (Top Left)
        airfoil_group = QGroupBox("1. Select Airfoil")
        airfoil_layout_inner = QVBoxLayout()
        self.xfoil_airfoil_list = QListWidget()
        self.xfoil_airfoil_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.xfoil_airfoil_list.itemSelectionChanged.connect(self.populate_reynolds_mach)
        airfoil_layout_inner.addWidget(self.xfoil_airfoil_list)
        airfoil_group.setLayout(airfoil_layout_inner)
        xfoil_layout.addWidget(airfoil_group, 0, 0) # Row 0, Col 0

        # Reynolds/Mach Selection (Middle Left)
        reynolds_mach_group = QGroupBox("2. Select Conditions")
        reynolds_mach_layout = QFormLayout() # Use Form layout for label/widget pairs
        self.xfoil_reynolds_combo = QComboBox()
        self.xfoil_mach_combo = QComboBox()
        reynolds_mach_layout.addRow("Reynolds Number:", self.xfoil_reynolds_combo)
        reynolds_mach_layout.addRow("Mach Number:", self.xfoil_mach_combo)
        reynolds_mach_group.setLayout(reynolds_mach_layout)
        xfoil_layout.addWidget(reynolds_mach_group, 1, 0) # Row 1, Col 0

        # Coefficient Selection (Bottom Left)
        coefficient_group = QGroupBox("3. Select Coefficients to Plot")
        coefficient_layout = QVBoxLayout()
        self.xfoil_cl_check = QCheckBox("Cl (Lift Coefficient)")
        self.xfoil_cd_check = QCheckBox("Cd (Drag Coefficient)")
        self.xfoil_cm_check = QCheckBox("Cm (Moment Coefficient)")
        self.xfoil_cl_check.setChecked(True) # Default check Cl
        coefficient_layout.addWidget(self.xfoil_cl_check)
        coefficient_layout.addWidget(self.xfoil_cd_check)
        coefficient_layout.addWidget(self.xfoil_cm_check)
        coefficient_group.setLayout(coefficient_layout)
        xfoil_layout.addWidget(coefficient_group, 2, 0) # Row 2, Col 0

        # Plot Button (Below selections)
        self.xfoil_results_plot_button = QPushButton("Plot Selected Results")
        self.xfoil_results_plot_button.clicked.connect(self.plot_xfoil_results)
        xfoil_layout.addWidget(self.xfoil_results_plot_button, 3, 0) # Row 3, Col 0

        # Plotting Area (Right side, spanning rows)
        self.xfoil_fig, self.xfoil_ax = plt.subplots()
        self.xfoil_canvas = FigureCanvas(self.xfoil_fig)
        self.xfoil_toolbar = NavigationToolbar(self.xfoil_canvas, self)
        xfoil_layout.addWidget(self.xfoil_toolbar, 0, 1, 1, 1) # Row 0, Col 1 (Toolbar)
        xfoil_layout.addWidget(self.xfoil_canvas, 1, 1, 3, 1) # Row 1-3, Col 1 (Canvas)

        # Adjust layout stretch factors
        xfoil_layout.setRowStretch(0, 2) # Give more space to airfoil list
        xfoil_layout.setRowStretch(1, 1)
        xfoil_layout.setRowStretch(2, 1)
        xfoil_layout.setRowStretch(3, 0) # Button row
        xfoil_layout.setColumnStretch(0, 1) # Selection column
        xfoil_layout.setColumnStretch(1, 3) # Plot column (wider)

        xfoil_tab.setLayout(xfoil_layout)
        self.tabs.addTab(xfoil_tab, "XFOIL Results Viewer") # Renamed tab slightly

    def populate_reynolds_mach(self):
        """Populates the Reynolds and Mach combo boxes based on selected airfoil."""
        self.xfoil_reynolds_combo.clear()
        self.xfoil_mach_combo.clear()
        # Clear plot when airfoil selection changes
        self.xfoil_ax.clear()
        self.xfoil_ax.set_title("Select Conditions and Coefficients to Plot")
        self.xfoil_canvas.draw()


        selected_items = self.xfoil_airfoil_list.selectedItems()
        if not selected_items:
            return # No airfoil selected

        airfoil_name = selected_items[0].text()

        if self.airfoil_db_instance:
            try:
                with sqlite3.connect(self.airfoil_db_instance.db_path) as conn:
                    cursor = conn.cursor()
                    # Query distinct Re/Mach pairs for the selected airfoil
                    cursor.execute("""SELECT DISTINCT reynolds_number, mach
                                      FROM aero_coeffs
                                      WHERE name = ?
                                      ORDER BY reynolds_number, mach""", (airfoil_name,))
                    results = cursor.fetchall()

                    if not results:
                        QMessageBox.information(self, "No Data", f"No XFOIL results found in the database for airfoil '{airfoil_name}'.")
                        return

                    reynolds_set = set()
                    mach_set = set()
                    for reynolds, mach in results:
                        if reynolds is not None:
                            reynolds_set.add(reynolds)
                        if mach is not None:
                            mach_set.add(mach)

                    # Add "All" options if needed, or just populate distinct values
                    # self.xfoil_reynolds_combo.addItem("All") # Example if needed
                    for reynolds in sorted(list(reynolds_set)):
                        self.xfoil_reynolds_combo.addItem(f"{reynolds:g}") # Format nicely
                    # self.xfoil_mach_combo.addItem("All") # Example if needed
                    for mach in sorted(list(mach_set)):
                         self.xfoil_mach_combo.addItem(f"{mach:.2f}") # Format nicely

            except sqlite3.Error as e:
                QMessageBox.critical(self, "Database Error", f"Error populating Reynolds/Mach numbers for '{airfoil_name}': {e}")
        else:
            QMessageBox.warning(self,"Database Error", "Database not loaded.")


    def plot_xfoil_results(self):
        """Plots the XFOIL results based on user selections."""
        if not self.airfoil_db_instance:
             QMessageBox.warning(self, "Plot Error", "Database not loaded.")
             return

        selected_items = self.xfoil_airfoil_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Plot XFOIL Results", "Please select an airfoil.")
            return
        airfoil_name = selected_items[0].text()

        # Get selected conditions (handle potential conversion errors)
        try:
            reynolds_str = self.xfoil_reynolds_combo.currentText()
            mach_str = self.xfoil_mach_combo.currentText()
            # Handle "All" or empty selections if implemented, otherwise convert
            reynolds = float(reynolds_str) if reynolds_str else None
            mach = float(mach_str) if mach_str else None
        except ValueError:
             QMessageBox.warning(self, "Plot Error", "Invalid Reynolds or Mach number selected.")
             return

        # Get selected coefficients
        coefficients_to_plot = []
        if self.xfoil_cl_check.isChecked(): coefficients_to_plot.append("cl")
        if self.xfoil_cd_check.isChecked(): coefficients_to_plot.append("cd")
        if self.xfoil_cm_check.isChecked(): coefficients_to_plot.append("cm")

        if not coefficients_to_plot:
            QMessageBox.warning(self, "Plot XFOIL Results", "Please select at least one coefficient (Cl, Cd, Cm) to plot.")
            return

        self.xfoil_ax.clear()
        plot_success = False
        try:
            # Fetch data for the specific Re and Mach
            data = self.airfoil_db_instance.get_aero_coeffs(airfoil_name, reynolds, mach)

            if not data:
                 QMessageBox.information(self, "No Data", f"No XFOIL data found for {airfoil_name} at Re={reynolds_str}, Mach={mach_str}.")
                 self.xfoil_ax.set_title(f"{airfoil_name} - No Data at Re={reynolds_str}, Mach={mach_str}")
                 self.xfoil_canvas.draw()
                 return

            # Prepare data for plotting (assuming data is sorted by alpha)
            alpha_values = [row[4] for row in data] # Index 4: alpha
            coeff_data = {
                "cl": [row[5] for row in data], # Index 5: cl
                "cd": [row[6] for row in data], # Index 6: cd
                "cm": [row[7] for row in data]  # Index 7: cm
            }

            for coeff in coefficients_to_plot:
                if coeff in coeff_data:
                    self.xfoil_ax.plot(alpha_values, coeff_data[coeff], marker='.', linestyle='-', label=f"{coeff.upper()}")
                    plot_success = True

            if plot_success:
                self.xfoil_ax.set_xlabel("Angle of Attack (Alpha, degrees)")
                self.xfoil_ax.set_ylabel("Coefficient Value")
                title = f"{airfoil_name} (Re={reynolds_str}, Mach={mach_str})"
                self.xfoil_ax.set_title(title)
                self.xfoil_ax.legend()
                self.xfoil_ax.grid(True)
            else:
                 # This case should ideally not happen if data was found
                 self.xfoil_ax.set_title(f"{airfoil_name} - Error Plotting Data")

            self.xfoil_canvas.draw()

        except sqlite3.Error as e:
            QMessageBox.critical(self, "Database Error", f"Error fetching XFOIL data: {e}")
            self.xfoil_ax.clear()
            self.xfoil_ax.set_title("Database Error")
            self.xfoil_canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Plotting Error", f"An unexpected error occurred during plotting: {e}")
            self.xfoil_ax.clear()
            self.xfoil_ax.set_title("Plotting Error")
            self.xfoil_canvas.draw()


    def setup_xfoil_results_search_tab(self):
        """Sets up the XFOIL results search tab."""
        xfoil_search_tab = QWidget()
        xfoil_search_layout = QGridLayout()

        # Input fields for XFOIL search parameters (Left side)
        param_group = QGroupBox("Search Criteria")
        param_layout = QFormLayout()

        self.xfoil_parameter_combo = QComboBox()
        # Use more descriptive names and map to DB columns if needed
        self.xfoil_parameter_combo.addItems(["Reynolds Number", "Mach Number", "Alpha", "Cl", "Cd", "Cm"])
        param_layout.addRow("Parameter:", self.xfoil_parameter_combo)

        self.xfoil_target_value_edit = QLineEdit()
        param_layout.addRow("Target Value:", self.xfoil_target_value_edit)

        self.xfoil_tolerance_edit = QLineEdit("0.1") # Default tolerance
        param_layout.addRow("Tolerance:", self.xfoil_tolerance_edit)

        self.xfoil_tolerance_type_combo = QComboBox()
        self.xfoil_tolerance_type_combo.addItems(["absolute", "percentage"])
        param_layout.addRow("Tolerance Type:", self.xfoil_tolerance_type_combo)

        param_group.setLayout(param_layout)
        xfoil_search_layout.addWidget(param_group, 0, 0) # Row 0, Col 0


        # Search button
        self.xfoil_search_button = QPushButton("Search XFOIL Results")
        self.xfoil_search_button.clicked.connect(self.perform_xfoil_results_search)
        xfoil_search_layout.addWidget(self.xfoil_search_button, 1, 0) # Row 1, Col 0

        # Clear Criteria Button
        self.xfoil_clear_criteria_button = QPushButton("Clear Criteria")
        self.xfoil_clear_criteria_button.clicked.connect(self.clear_xfoil_search_criteria)
        xfoil_search_layout.addWidget(self.xfoil_clear_criteria_button, 2, 0) # Row 2, Col 0


        xfoil_search_layout.setRowStretch(3, 1) # Add stretch below inputs

        # Results list (Middle column)
        xfoil_search_layout.addWidget(QLabel("Matching Airfoils:"), 0, 1) # Label
        self.xfoil_search_results_list = QListWidget()
        self.xfoil_search_results_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        xfoil_search_layout.addWidget(self.xfoil_search_results_list, 1, 1, 3, 1) # Span rows 1-3, Col 1

        # Clear List Button (Below list)
        self.xfoil_search_clear_list_button = QPushButton("Clear List")
        self.xfoil_search_clear_list_button.clicked.connect(self.clear_xfoil_search_list)
        xfoil_search_layout.addWidget(self.xfoil_search_clear_list_button, 4, 1) # Row 4, Col 1


        # Plot area (Right column)
        self.xfoil_search_fig, self.xfoil_search_ax = plt.subplots()
        self.xfoil_search_canvas = FigureCanvas(self.xfoil_search_fig)
        self.xfoil_search_toolbar = NavigationToolbar(self.xfoil_search_canvas, self)
        xfoil_search_layout.addWidget(self.xfoil_search_toolbar, 0, 2) # Row 0, Col 2
        xfoil_search_layout.addWidget(self.xfoil_search_canvas, 1, 2, 3, 1) # Span rows 1-3, Col 2

        # Plot buttons (Below plot)
        self.xfoil_clear_plot_button = QPushButton("Clear Plot")
        self.xfoil_clear_plot_button.clicked.connect(self.clear_xfoil_search_plot)
        xfoil_search_layout.addWidget(self.xfoil_clear_plot_button, 4, 2) # Row 4, Col 2

        self.xfoil_plot_selected_button = QPushButton("Plot Selected Airfoils")
        self.xfoil_plot_selected_button.clicked.connect(self.plot_selected_airfoils_xfoil_search_tab)
        xfoil_search_layout.addWidget(self.xfoil_plot_selected_button, 5, 2) # Row 5, Col 2

        # Column stretches
        xfoil_search_layout.setColumnStretch(0, 1) # Criteria
        xfoil_search_layout.setColumnStretch(1, 2) # Results List
        xfoil_search_layout.setColumnStretch(2, 3) # Plot

        xfoil_search_tab.setLayout(xfoil_search_layout)
        self.tabs.addTab(xfoil_search_tab, "XFOIL Results Search")

    def perform_xfoil_results_search(self):
        """Performs the XFOIL results search."""
        if not self.airfoil_db_instance:
             QMessageBox.warning(self, "Search Error", "Database not loaded.")
             return

        # Map display names to potential database column names if different
        parameter_map = {
            "Reynolds Number": "reynolds_number", # Adjust if DB column name is different
            "Mach Number": "mach",
            "Alpha": "alpha",
            "Cl": "cl",
            "Cd": "cd",
            "Cm": "cm"
        }
        parameter_display_name = self.xfoil_parameter_combo.currentText()
        parameter = parameter_map.get(parameter_display_name, parameter_display_name.lower()) # Fallback to lower case


        # Suggestion 7: Validate numeric inputs
        try:
            target_value_str = self.xfoil_target_value_edit.text().strip()
            tolerance_str = self.xfoil_tolerance_edit.text().strip()

            if not target_value_str:
                 raise ValueError("Target value cannot be empty")
            if not tolerance_str:
                 raise ValueError("Tolerance cannot be empty")

            target_value = float(target_value_str)
            tolerance = float(tolerance_str)
            if tolerance < 0:
                 raise ValueError("Tolerance cannot be negative")

        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid numeric input for Target Value or Tolerance: {e}. Please correct it.")
            return

        tolerance_type = self.xfoil_tolerance_type_combo.currentText()

        # Clear previous results
        self.xfoil_search_results_list.clear()
        self.clear_xfoil_search_plot() # Clear plot too

        try:
            results = self.airfoil_db_instance.find_airfoils_by_xfoil_results(
                parameter, target_value, tolerance, tolerance_type
            )

            if results:
                unique_results = sorted(list(set(results))) # Ensure unique and sort
                self.xfoil_search_results_list.addItems(unique_results)
                QMessageBox.information(self, "Search Complete", f"Found {len(unique_results)} airfoil(s) matching the criteria.")
            else:
                QMessageBox.information(self, "Search Complete", "No matching airfoils found for the specified XFOIL criteria.")

        except AttributeError:
             # Handle case where the backend method might not exist
             QMessageBox.critical(self, "Search Error", "The backend function 'find_airfoils_by_xfoil_results' is not available.")
        except Exception as e:
             QMessageBox.critical(self, "Search Error", f"An error occurred during XFOIL results search: {e}")


    def clear_xfoil_search_criteria(self):
        """Clears the input fields for the XFOIL results search."""
        self.xfoil_parameter_combo.setCurrentIndex(0)
        self.xfoil_target_value_edit.clear()
        self.xfoil_tolerance_edit.setText("0.1") # Reset default
        self.xfoil_tolerance_type_combo.setCurrentIndex(0)

    def clear_xfoil_search_list(self):
        """Clears the XFOIL search results list."""
        self.xfoil_search_results_list.clear()

    def clear_xfoil_search_plot(self):
        """Clears the XFOIL search plot."""
        self.xfoil_search_ax.clear()
        self.xfoil_search_ax.set_title("Airfoil Plot Area")
        self.xfoil_search_ax.set_xlabel("X Coordinate")
        self.xfoil_search_ax.set_ylabel("Y Coordinate")
        self.xfoil_search_ax.grid(False)
        self.xfoil_search_canvas.draw()

    def plot_selected_airfoils_xfoil_search_tab(self):
        """Plots the selected airfoils from the XFOIL search results list."""
        if not self.airfoil_db_instance: return
        selected_items = self.xfoil_search_results_list.selectedItems()
        names = [item.text() for item in selected_items]
        if names:
            self.xfoil_search_ax.clear()
            plotted_count = 0
            max_plots = 10 # Limit plots
            if len(names) > max_plots:
                 QMessageBox.information(self, "Plot Limit", f"Plotting the first {max_plots} selected airfoils.")
                 names = names[:max_plots]

            for name in names:
                try:
                    self.airfoil_db_instance.add_airfoil_to_plot(name, self.xfoil_search_ax, linestyle='-', marker=None, markersize=3)
                    plotted_count += 1
                except Exception as e:
                     print(f"Warning: Could not plot airfoil '{name}' from XFOIL search results: {e}")

            if plotted_count > 0:
                self.xfoil_search_ax.set_xlabel("X Coordinate")
                self.xfoil_search_ax.set_ylabel("Y Coordinate")
                self.xfoil_search_ax.set_title("Selected Airfoil Comparison")
                self.xfoil_search_ax.grid(True)
                self.xfoil_search_ax.axis('equal')
                self.xfoil_search_ax.legend()
                self.xfoil_search_canvas.draw()
            else:
                 self.clear_xfoil_search_plot()
                 QMessageBox.warning(self, "Plot Error", "Could not plot any of the selected airfoils.")
        else:
            QMessageBox.information(self, "Plot Selected", "Please select one or more airfoils from the results list to plot.")

    # Suggestion 2: Removed setup_xfoil_plot method
    # Suggestion 2: Removed perform_xfoil_plot method


class PointEditDialog(QDialog):
    """Simple dialog to edit X and Y coordinates."""
    def __init__(self, x, y, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Point Coordinates")
        layout = QFormLayout()

        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setDecimals(6) # More precision
        self.x_spinbox.setRange(-1e6, 1e6) # Set reasonable range
        self.x_spinbox.setValue(x)
        layout.addRow("X:", self.x_spinbox)

        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setDecimals(6) # More precision
        self.y_spinbox.setRange(-1e6, 1e6) # Set reasonable range
        self.y_spinbox.setValue(y)
        layout.addRow("Y:", self.y_spinbox)

        # OK and Cancel buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(ok_button)
        button_box.addWidget(cancel_button)
        layout.addRow(button_box) # Add buttons row

        self.setLayout(layout)

    def get_values(self):
        """Returns the edited X and Y values."""
        return self.x_spinbox.value(), self.y_spinbox.value()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply styles or settings if desired
    # app.setStyle('Fusion')
    viewer = AirfoilViewer()
    viewer.show()
    sys.exit(app.exec())