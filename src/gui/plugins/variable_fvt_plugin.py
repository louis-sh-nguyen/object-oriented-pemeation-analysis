import os
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from . import ModelPlugin, ModeFrame
from src.utils.defaults import DEFAULTS, THICKNESS_DICT, FLOWRATE_DICT, FVT_FITTING_DEFAULTS

class VariableFVTManual(ModeFrame):
    def __init__(self, parent):
        # Initialize instance variables before parent's __init__
        self.parameter_entries = {}
        self.status_label = None
        self.progress = None
        self.results_text = None
        self.fig = None
        self.axes = None
        self.canvas = None
        
        # Call parent's __init__ after initializing our variables
        super().__init__(parent, "Manual", "Variable FVT")
        
    def setup_input_content(self):
        """Set up manual mode input tab"""
        # Parameters Box for all inputs
        params_box = ctk.CTkFrame(self.input_scroll)
        params_box.pack(fill="x", padx=10, pady=5)
        
        # Parameters Box Title
        ctk.CTkLabel(params_box, text="Model Parameters", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
        
        # Parameter frame
        self.create_parameter_frame(params_box)
        
        # Generate button
        ctk.CTkButton(self.input_scroll, 
                     text="Generate Results",
                     command=self.calculate_results).pack(pady=10)

    def create_parameter_frame(self, parent):
        """Create parameter inputs frame"""
        param_frame = ctk.CTkFrame(parent)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Parameter definitions
        parameters = [
            ("Required Parameters", [
                ("thickness", "Thickness (mm)", True, str(DEFAULTS["thickness"])),
                ("diameter", "Diameter (cm)", True, str(DEFAULTS["diameter"])),
                ("flowrate", "Flow Rate (cm³/min)", True, str(DEFAULTS["flowrate"])),
            ]), 
            ("Optional Parameters", [
                ("pressure", "Pressure (bar)", False, ""),
                ("temperature", "Temperature (°C)", False, ""),
                ("D1_prime", "D1 Prime", False, str(FVT_FITTING_DEFAULTS["D1_prime"]["initial"])),
                ("DT0", "DT0", False, str(FVT_FITTING_DEFAULTS["DT0"]["initial"]))
            ])
        ]
        
        # Create sections
        for section_title, params in parameters:
            # Section title
            ctk.CTkLabel(param_frame, 
                        text=section_title,
                        font=ctk.CTkFont(weight="bold")).pack(pady=5)
            
            # Parameters
            for param_name, param_label, required, default_value in params:
                param_row = ctk.CTkFrame(param_frame)
                param_row.pack(fill="x", padx=5, pady=2)
                
                # Label
                label_text = f"{param_label}{'*' if required else ''}"
                ctk.CTkLabel(param_row, text=label_text).pack(side="left", padx=5)
                
                # Entry
                entry = ctk.CTkEntry(param_row, width=120)
                if default_value:
                    entry.insert(0, default_value)
                entry.pack(side="right", padx=5)
                
                # Store entry reference
                self.parameter_entries[param_name] = entry

    def setup_results_content(self):
        """Set up results display"""
        # Progress frame
        progress_frame = ctk.CTkFrame(self.results_frame)
        progress_frame.pack(fill="x", pady=5, padx=10)
        
        # Status label
        self.status_label = ctk.CTkLabel(progress_frame, text="Ready")
        self.status_label.pack(side="left", padx=5)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(progress_frame)
        self.progress.pack(side="left", fill="x", expand=True, padx=5)
        self.progress.set(0)
        
        # Results text
        self.results_text = ctk.CTkTextbox(self.results_frame, height=100)
        self.results_text.pack(fill="x", pady=5, padx=10)
        
        # Plot frame
        plot_frame = ctk.CTkFrame(self.results_frame)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create figure with subplots
        self.fig = Figure(figsize=(12, 10))
        self.axes = {
            'diffusivity_profile': self.fig.add_subplot(221),
            'diffusivity_location': self.fig.add_subplot(222),
            'flux_time': self.fig.add_subplot(223),
            'flux_tau': self.fig.add_subplot(224)
        }
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        
        # Add resize handler
        def on_resize(event):
            w, h = event.width, event.height
            self.fig.set_size_inches(w/self.fig.get_dpi(), h/self.fig.get_dpi())
            self.fig.tight_layout()
            self.canvas.draw()
        
        canvas_widget.bind('<Configure>', on_resize)

    def get_parameters(self):
        """Get parameters from entries"""
        params = {}
        for name, entry in self.parameter_entries.items():
            value = entry.get()
            if value:  # Only include non-empty values
                try:
                    params[name] = float(value)
                except ValueError:
                    self.show_error(f"Invalid value for {name}")
                    return None
        return params

    def show_error(self, message):
        """Show error message"""
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", f"Error: {message}")

    def calculate_results(self):
        """Execute manual calculation workflow"""
        self.tabview.set("Results")  # Switch to results tab
        
        # Get parameters
        params = self.get_parameters()
        if not params:
            return
            
        # Verify required parameters
        required_params = ['thickness', 'diameter', 'flowrate', 'D1_prime', 'DT0']
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            self.show_error(f"Missing required parameters: {', '.join(missing_params)}")
            return
            
        # TODO: Add actual calculation logic here
        self.status_label.configure(text="Calculation complete")
        self.progress.set(1.0)

class VariableFVTFitting(ModeFrame):
    def __init__(self, parent):
        # Initialize instance variables
        self.parameter_entries = {}
        self.file_source = None
        self.file_selector = None
        self.file_entry = None
        self.stab_threshold = None
        self.preset_frame = None
        self.browse_frame = None
        self.fit_option = None
        self.dt0_settings = None
        self.bounds_entries = {}
        
        super().__init__(parent, "Fitting", "Variable FVT")

    def setup_input_content(self):
        """Set up fitting mode input tab"""
        # Title and description
        title_frame = ctk.CTkFrame(self.input_scroll)
        title_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(title_frame, 
                    text="Variable Diffusivity Model - Fitting", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)

        # Create sections
        self.create_membrane_params()
        self.create_experimental_params()
        self.create_data_selection()
        self.create_fitting_params()
        
        # Fit button
        ctk.CTkButton(self.input_scroll,
                     text="Start Fitting",
                     command=self.start_fitting,
                     height=40).pack(pady=20)

    def create_membrane_params(self):
        """Create membrane parameters section"""
        frame = self.create_section_frame("Membrane Parameters")
        
        params = [
            ("thickness", "Thickness (mm)*", str(DEFAULTS["thickness"])),
            ("diameter", "Diameter (cm)*", str(DEFAULTS["diameter"]))
        ]
        
        self.create_param_entries(frame, params)

    def create_experimental_params(self):
        """Create experimental parameters section"""
        frame = self.create_section_frame("Experimental Parameters")
        
        params = [
            ("flowrate", "Flow Rate (cm³/min)*", str(DEFAULTS["flowrate"])),
            ("pressure", "Pressure (bar)", ""),
            ("temperature", "Temperature (°C)", "")
        ]
        
        self.create_param_entries(frame, params)

    def create_data_selection(self):
        """Create data selection section"""
        frame = self.create_section_frame("Data Selection")
        
        # Data source selection
        source_frame = ctk.CTkFrame(frame)
        source_frame.pack(fill="x", pady=5)
        
        # Label (right-aligned)
        label_frame = ctk.CTkFrame(source_frame)
        label_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(label_frame, text="Data Source:").pack(side="right", padx=5)
        
        # Radio buttons
        buttons_frame = ctk.CTkFrame(source_frame)
        buttons_frame.pack(side="right", padx=10)
        
        self.file_source = ctk.StringVar(value="preset")
        ctk.CTkRadioButton(buttons_frame, text="Preset", 
                          variable=self.file_source, value="preset",
                          command=self.toggle_file_source).pack(side="left", padx=5)
        ctk.CTkRadioButton(buttons_frame, text="Browse", 
                          variable=self.file_source, value="browse",
                          command=self.toggle_file_source).pack(side="left", padx=5)
        
        # File selection frames
        self.preset_frame = self.create_preset_frame(frame)
        self.browse_frame = self.create_browse_frame(frame)
        
        # Stabilisation threshold
        thresh_frame = ctk.CTkFrame(frame)
        thresh_frame.pack(fill="x", pady=5)
        
        label_frame = ctk.CTkFrame(thresh_frame)
        label_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(label_frame, text="Stabilisation Threshold:").pack(side="right", padx=5)
        
        self.stab_threshold = ctk.CTkEntry(thresh_frame, width=120)
        self.stab_threshold.insert(0, "0.002")
        self.stab_threshold.pack(side="right", padx=10)

    def create_fitting_params(self):
        """Create fitting parameters section"""
        frame = self.create_section_frame("Fitting Parameters")
        
        # Fit mode selection
        mode_frame = ctk.CTkFrame(frame)
        mode_frame.pack(fill="x", pady=5)
        
        label_frame = ctk.CTkFrame(mode_frame)
        label_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(label_frame, text="Fitting Mode:").pack(side="right", padx=5)
        
        self.fit_option = ctk.CTkComboBox(
            mode_frame,
            values=["D1 Prime Only", "D1 Prime & DT0"],
            command=self.update_fitting_inputs,
            state="readonly",
            width=150
        )
        self.fit_option.pack(side="right", padx=10)
        self.fit_option.set("D1 Prime Only")
        
        # Create bounds frames
        self.create_bounds_section(frame, "D1 Prime", 
                                 FVT_FITTING_DEFAULTS["D1_prime"])
        
        # DT0 settings (initially hidden)
        self.dt0_settings = ctk.CTkFrame(frame)
        self.create_bounds_section(self.dt0_settings, "DT0", 
                                 FVT_FITTING_DEFAULTS["DT0"])
        
        # Number of starts
        starts_frame = ctk.CTkFrame(frame)
        starts_frame.pack(fill="x", pady=5)
        
        label_frame = ctk.CTkFrame(starts_frame)
        label_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(label_frame, text="Number of Starts:").pack(side="right", padx=5)
        
        self.n_starts = ctk.CTkEntry(starts_frame, width=120)
        self.n_starts.insert(0, str(FVT_FITTING_DEFAULTS["n_starts"]))
        self.n_starts.pack(side="right", padx=10)

    def create_bounds_section(self, parent, name, defaults):
        """Create parameter bounds and initial value inputs"""
        # Bounds
        bounds_frame = ctk.CTkFrame(parent)
        bounds_frame.pack(fill="x", pady=5)
        
        label_frame = ctk.CTkFrame(bounds_frame)
        label_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(label_frame, text=f"{name} Bounds:").pack(side="right", padx=5)
        
        entry_frame = ctk.CTkFrame(bounds_frame)
        entry_frame.pack(side="right", padx=10)
        
        lower = ctk.CTkEntry(entry_frame, width=80, placeholder_text="Lower")
        lower.insert(0, str(defaults["lower_bound"]))
        lower.pack(side="left", padx=2)
        
        ctk.CTkLabel(entry_frame, text="→").pack(side="left", padx=2)
        
        upper = ctk.CTkEntry(entry_frame, width=80, placeholder_text="Upper")
        upper.insert(0, str(defaults["upper_bound"]))
        upper.pack(side="left", padx=2)
        
        # Initial value
        init_frame = ctk.CTkFrame(parent)
        init_frame.pack(fill="x", pady=2)
        
        label_frame = ctk.CTkFrame(init_frame)
        label_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(label_frame, text=f"{name} Initial:").pack(side="right", padx=5)
        
        initial = ctk.CTkEntry(init_frame, width=120)
        initial.insert(0, str(defaults["initial"]))
        initial.pack(side="right", padx=10)
        
        # Store references
        key = name.lower().replace(" ", "")
        self.bounds_entries[key] = {
            "lower": lower,
            "upper": upper,
            "initial": initial
        }

    def create_section_frame(self, title):
        """Create a section frame with title"""
        section = ctk.CTkFrame(self.input_scroll)
        section.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(section, text=title,
                    font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        separator = ctk.CTkFrame(section, height=1)
        separator.pack(fill="x", padx=10)
        
        return section

    def create_param_entries(self, parent, params):
        """Create parameter entries in a section"""
        for param_name, label_text, default_value in params:
            param_row = ctk.CTkFrame(parent)
            param_row.pack(fill="x", padx=10, pady=5)
            
            # Label (right-aligned)
            label_frame = ctk.CTkFrame(param_row)
            label_frame.pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(label_frame, text=label_text).pack(side="right", padx=5)
            
            # Entry (fixed width)
            entry = ctk.CTkEntry(param_row, width=120)
            entry.pack(side="right", padx=10)
            
            if default_value:
                entry.insert(0, default_value)
                
            self.parameter_entries[param_name] = entry

    def setup_results_content(self):
        """Set up fitting mode results tab"""
        # Add basic results display elements
        self.results_label = ctk.CTkLabel(self.results_frame, 
                                        text="No fitting results yet")
        self.results_label.pack(pady=10)
        
        # Add simple plot
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10, expand=True, fill="both")
    
    def start_fitting(self):
        """Execute fitting workflow"""
        # Switch to results tab
        self.tabview.set("Results")
        
        # TODO: Add actual fitting logic here
        self.results_label.configure(text="Fitting complete")

    def get_data_files(self):
        """Get list of Excel files in data directory"""
        data_dir = os.path.join("data", "single_pressure")
        try:
            files = [f for f in os.listdir(data_dir) 
                    if f.endswith(('.xlsx', '.xls'))]
            return files
        except Exception as e:
            print(f"Error reading data directory: {e}")
            return []

    def toggle_file_source(self):
        """Toggle between preset files and browse options"""
        if self.file_source.get() == "preset":
            self.browse_frame.pack_forget()
            self.preset_frame.pack(fill="x")
        else:
            self.preset_frame.pack_forget()
            self.browse_frame.pack(fill="x")

    def browse_file(self):
        """Open file browser dialog"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, filename)

    def update_fitting_inputs(self, mode):
        """Update visible fitting inputs based on selected mode"""
        if mode == "D1 Prime Only":
            self.dt0_settings.pack_forget()
        else:
            self.dt0_settings.pack(fill="x", pady=5)

    def create_preset_frame(self, parent):
        """Create frame for preset file selection"""
        preset_frame = ctk.CTkFrame(parent)
        preset_frame.pack(fill="x", pady=5)
        
        # Label (right-aligned)
        label_frame = ctk.CTkFrame(preset_frame)
        label_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(label_frame, text="Preset File:").pack(side="right", padx=5)
        
        # Dropdown and refresh button
        control_frame = ctk.CTkFrame(preset_frame)
        control_frame.pack(side="right", padx=10)
        
        self.file_selector = ctk.CTkComboBox(
            control_frame,
            values=self.get_data_files(),
            width=200,
            state="readonly"
        )
        self.file_selector.pack(side="left", padx=2)
        
        ctk.CTkButton(control_frame, text="↻", width=30,
                     command=self.refresh_files).pack(side="left", padx=2)
        
        return preset_frame

    def create_browse_frame(self, parent):
        """Create frame for file browser"""
        browse_frame = ctk.CTkFrame(parent)
        browse_frame.pack(fill="x", pady=5)
        
        # Label (right-aligned)
        label_frame = ctk.CTkFrame(browse_frame)
        label_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(label_frame, text="File Path:").pack(side="right", padx=5)
        
        # Entry and browse button
        control_frame = ctk.CTkFrame(browse_frame)
        control_frame.pack(side="right", padx=10)
        
        self.file_entry = ctk.CTkEntry(control_frame, width=200)
        self.file_entry.pack(side="left", padx=2)
        
        ctk.CTkButton(control_frame, text="Browse",
                     command=self.browse_file).pack(side="left", padx=2)
        
        return browse_frame

    def refresh_files(self):
        """Refresh the list of data files in the dropdown"""
        current = self.file_selector.get()
        files = self.get_data_files()
        self.file_selector.configure(values=files)
        
        # Try to keep the current selection if it still exists
        if current in files:
            self.file_selector.set(current)
        elif files:
            self.file_selector.set(files[0])

class VariableFVTPlugin(ModelPlugin):
    def __init__(self, parent):
        super().__init__(parent)
    
    def setup_frames(self):
        """Create manual and fitting frames"""
        self.manual_frame = VariableFVTManual(self.parent)
        self.fitting_frame = VariableFVTFitting(self.parent)  # Fixed the typo 'fittime' to 'fitting_frame'
