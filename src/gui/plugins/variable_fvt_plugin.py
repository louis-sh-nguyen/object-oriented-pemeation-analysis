import os
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from . import ModelPlugin, ModeFrame
from src.utils.defaults import DEFAULTS, THICKNESS_DICT, FLOWRATE_DICT, FVT_FITTING_DEFAULTS
from src.models.single_pressure.variable_diffusivity_fvt.workflow import manual_workflow, data_fitting_workflow
from src.models.single_pressure.variable_diffusivity_fvt.plotting import (
    plot_diffusivity_profile,
    plot_diffusivity_location_profile,
    plot_norm_flux_over_time,
    plot_norm_flux_over_tau
)

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
        
        # Simulation Settings Title
        ctk.CTkLabel(params_box, text="Simulation Settings", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
        
        # Simulation settings frame
        self.create_simulation_settings_frame(params_box)
        
        # Generate button
        ctk.CTkButton(self.input_scroll, 
                     text="Generate Results",
                     command=self.run_results).pack(pady=10)

    def create_parameter_frame(self, parent):
        """Create parameter inputs frame"""
        param_frame = ctk.CTkFrame(parent)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Parameter definitions
        parameters = [
            ("Required", [
                ("thickness", "Thickness (mm)", True, str(DEFAULTS["thickness"])),
                ("diameter", "Diameter (cm)", True, str(DEFAULTS["diameter"])),
                ("flowrate", "Flow Rate (cm³/min)", True, str(DEFAULTS["flowrate"])),
            ]), 
            ("Optional", [
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

    def create_simulation_settings_frame(self, parent):
        """Create simulation settings frame"""
        sim_frame = ctk.CTkFrame(parent)
        sim_frame.pack(fill="x", padx=10, pady=5)
        
        # Parameter entries dictionary for simulation settings
        self.sim_entries = {}
        
        # Parameters
        parameters = [
            ("T", "Simulation Time (s)", "10000"),
            ("dx", "Spatial Step Size", "0.005")
        ]
        
        # Create entries
        for param_name, param_label, default_value in parameters:
            param_row = ctk.CTkFrame(sim_frame)
            param_row.pack(fill="x", padx=5, pady=2)
            
            # Label
            ctk.CTkLabel(param_row, text=param_label).pack(side="left", padx=5)
            
            # Entry
            entry = ctk.CTkEntry(param_row, width=120)
            entry.insert(0, default_value)
            entry.pack(side="right", padx=5)
            
            # Store entry reference
            self.sim_entries[param_name] = entry

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
        # Get model parameters
        for name, entry in self.parameter_entries.items():
            value = entry.get()
            if value:  # Only include non-empty values
                try:
                    params[name] = float(value)
                except ValueError:
                    self.show_error(f"Invalid value for {name}")
                    return None
        
        # Get simulation parameters
        params['simulation'] = {}
        for name, entry in self.sim_entries.items():
            value = entry.get()
            if value:
                try:
                    params['simulation'][name] = float(value)
                except ValueError:
                    self.show_error(f"Invalid value for simulation parameter {name}")
                    return None
        
        return params

    def show_error(self, message):
        """Show error message"""
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", f"Error: {message}")

    def run_results(self):
        """Execute manual calculation workflow"""
        self.tabview.set("Results")  # Switch to results tab
        
        # Get parameters
        params = self.get_parameters()
        if not params:
            return
            
        # Verify all required parameters
        required_model_params = ['thickness', 'diameter', 'flowrate', 'D1_prime', 'DT0']
        required_sim_params = ['T', 'dx']
        
        missing_model_params = [p for p in required_model_params if p not in params]
        missing_sim_params = [p for p in required_sim_params if p not in params['simulation']]
        
        if missing_model_params or missing_sim_params:
            error_msg = []
            if missing_model_params:
                error_msg.append(f"Missing model parameters: {', '.join(missing_model_params)}")
            if missing_sim_params:
                error_msg.append(f"Missing simulation parameters: {', '.join(missing_sim_params)}")
            self.show_error("\n".join(error_msg))
            return

        try:
            self.status_label.configure(text="Running calculations...")
            self.progress.set(0.2)
            
            # Run manual workflow with parameters
            model, Dprime_df, flux_df, figures = manual_workflow(
                pressure=params.get('pressure', 1.0),
                temperature=params.get('temperature', 25.0),
                thickness=params['thickness'],
                diameter=params['diameter'],
                flowrate=params['flowrate'],
                D1_prime=params['D1_prime'],
                DT_0=params['DT0'],
                simulation_params={
                    'T': params['simulation']['T'],
                    'dx': params['simulation']['dx'],
                    'X': 1.0
                },
                output_settings={
                    'display_plots': False,
                    'save_plots': False,
                    'save_data': False
                }
            )
            
            self.progress.set(0.6)
            
            # Clear the entire figure and create new subplots
            self.fig.clear()
            self.axes = {
                'diffusivity_profile': self.fig.add_subplot(221),
                'diffusivity_location': self.fig.add_subplot(222),
                'flux_time': self.fig.add_subplot(223),
                'flux_tau': self.fig.add_subplot(224)
            }
            
            # Display results text
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", "Manual Calculation Results:\n")
            self.results_text.insert("end", f"D1 Prime: {params['D1_prime']:.4e}\n")
            self.results_text.insert("end", f"DT0: {params['DT0']:.4e}\n")
            
            self.progress.set(0.8)
            
            # Plot results in the four subplots with explicit figure reference
            plot_diffusivity_profile(
                diffusivity_profile=Dprime_df,
                ax=self.axes['diffusivity_profile'],
                fig=self.fig,
                display=False
            )
            
            plot_diffusivity_location_profile(
                diffusivity_profile=Dprime_df,
                L=params['thickness'],
                T=flux_df['time'].max(),
                ax=self.axes['diffusivity_location'],
                fig=self.fig,
                display=False
            )
            
            plot_norm_flux_over_time(
                flux_data=flux_df,
                ax=self.axes['flux_time'],
                fig=self.fig,
                display=False
            )
            
            plot_norm_flux_over_tau(
                flux_data=flux_df,
                ax=self.axes['flux_tau'],
                fig=self.fig,
                display=False
            )
            
            # Update plot layout and display
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Final status update
            self.status_label.configure(text="Complete")
            self.progress.set(1.0)
            
        except Exception as e:
            self.show_error(f"Calculation error: {str(e)}")
            self.status_label.configure(text="Error")
            self.progress.set(0)

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
        # Parameters Box for all inputs
        params_box = ctk.CTkFrame(self.input_scroll)
        params_box.pack(fill="x", padx=10, pady=5)
        
        # Parameters Box Title
        ctk.CTkLabel(params_box, text="Model Parameters", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
        
        # Create parameter frame
        self.create_parameter_frame(params_box)
        
        # Data Selection Title
        ctk.CTkLabel(params_box, text="Data Selection", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
        
        # Add file selection section
        self.create_file_selection(params_box)
        
        # Fitting Settings Title
        ctk.CTkLabel(params_box, text="Fitting Settings", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
        
        # Add fitting options section
        self.create_fitting_options(params_box)
        
        # Fit button
        ctk.CTkButton(self.input_scroll,
                     text="Start Fitting",
                     command=self.run_results).pack(pady=10)

    def create_parameter_frame(self, parent):
        """Create parameter inputs frame"""
        param_frame = ctk.CTkFrame(parent)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Parameter definitions
        parameters = [
            ("Required", [
                ("thickness", "Thickness (mm)", True, str(DEFAULTS["thickness"])),
                ("diameter", "Diameter (cm)", True, str(DEFAULTS["diameter"])),
                ("flowrate", "Flow Rate (cm³/min)", True, str(DEFAULTS["flowrate"]))
            ]),
            ("Optional", [
                ("pressure", "Pressure (bar)", False, ""),
                ("temperature", "Temperature (°C)", False, ""),
                ("D1_prime", "D1'", False, str(FVT_FITTING_DEFAULTS["D1_prime"]["initial"])),
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

    def create_file_selection(self, parent):
        """Create file selection and stabilisation threshold frame"""
        # Define consistent width for file selection controls
        FILE_SELECT_WIDTH = 250

        file_frame = ctk.CTkFrame(parent)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        # Title for section
        # ctk.CTkLabel(file_frame, text="Data Selection", 
        #             font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        # Data source selection
        source_frame = ctk.CTkFrame(file_frame)
        source_frame.pack(fill="x", pady=5)
        
        # Radio buttons for file source
        self.file_source = ctk.StringVar(value="preset")
        ctk.CTkRadioButton(source_frame, text="Preset Files", 
                          variable=self.file_source, value="preset",
                          command=self.toggle_file_source).pack(side="left", padx=5)
        ctk.CTkRadioButton(source_frame, text="Browse", 
                          variable=self.file_source, value="browse",
                          command=self.toggle_file_source).pack(side="left", padx=5)
        
        # File selection container
        self.file_select_container = ctk.CTkFrame(file_frame)
        self.file_select_container.pack(fill="x", pady=5)
        
        # Preset files dropdown
        self.preset_frame = ctk.CTkFrame(self.file_select_container)
        ctk.CTkLabel(self.preset_frame, text="Select File:").pack(side="left", padx=5)
        self.file_selector = ctk.CTkComboBox(
            self.preset_frame,
            values=self.get_data_files(),
            width=FILE_SELECT_WIDTH,
            state="readonly",
            command=self.on_file_selected  # Set callback directly
        )
        self.file_selector.pack(side="left", padx=5)
        ctk.CTkButton(self.preset_frame, text="↻", width=30,
                     command=self.refresh_files).pack(side="left", padx=5)
        
        # Browse file frame
        self.browse_frame = ctk.CTkFrame(self.file_select_container)
        ctk.CTkLabel(self.browse_frame, text="Select File:").pack(side="left", padx=5)
        self.file_entry = ctk.CTkEntry(
            self.browse_frame, 
            width=FILE_SELECT_WIDTH
        )
        self.file_entry.pack(side="left", padx=5)
        ctk.CTkButton(self.browse_frame, text="Browse", 
                     command=self.browse_file).pack(side="left", padx=5)

        # Stabilisation threshold frame
        stab_frame = ctk.CTkFrame(file_frame)
        stab_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(stab_frame, text="Stabilisation Threshold:").pack(side="left", padx=5)
        self.stab_threshold = ctk.CTkEntry(stab_frame, width=100, placeholder_text="0.002")
        self.stab_threshold.pack(side="left", padx=5)
        
        # Add info tooltip/label
        ctk.CTkLabel(stab_frame, 
                    text="(0.005 for breakthrough, 0.002 for full curve)", 
                    text_color="gray").pack(side="left", padx=5)

        # Show preset frame by default
        self.preset_frame.pack(fill="x")

        # Set default stabilisation threshold
        self.stab_threshold.insert(0, "0.002")

    def create_fitting_options(self, parent):
        """Create fitting options section"""
        # Create main frame for all fitting options
        fitting_frame = ctk.CTkFrame(parent)
        fitting_frame.pack(fill="x", padx=10, pady=5)

        # Main parameters grid frame
        options_frame = ctk.CTkFrame(fitting_frame)
        options_frame.pack(fill="x", padx=10, pady=5)
        options_frame.grid_columnconfigure(1, weight=1)
        
        row = 0
        # Fit mode selector
        ctk.CTkLabel(options_frame, text="Mode:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.fit_option = ctk.CTkComboBox(
            options_frame,
            values=["D1' Only", "D1' & DT0"],
            command=self.update_fitting_inputs,
            state="readonly",
            width=120
        )
        self.fit_option.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.fit_option.set("D1' Only")

        # D1 Prime settings
        row += 1
        ctk.CTkLabel(options_frame, text="D1' Initial:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.d1_initial = ctk.CTkEntry(options_frame, width=100)
        self.d1_initial.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        row += 1
        ctk.CTkLabel(options_frame, text="D1' Bounds:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        d1_bounds_frame = ctk.CTkFrame(options_frame)
        d1_bounds_frame.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.d1_lower = ctk.CTkEntry(d1_bounds_frame, width=100, placeholder_text="Lower")
        self.d1_lower.pack(side="left", padx=5)
        ctk.CTkLabel(d1_bounds_frame, text="→").pack(side="left", padx=5)
        self.d1_upper = ctk.CTkEntry(d1_bounds_frame, width=100, placeholder_text="Upper")
        self.d1_upper.pack(side="left", padx=5)

        # DT0 settings
        row += 1
        self.dt0_initial_label = ctk.CTkLabel(options_frame, text="DT0 Initial:")
        self.dt0_initial_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.dt0_initial = ctk.CTkEntry(options_frame, width=100)
        self.dt0_initial.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        row += 1
        self.dt0_bounds_label = ctk.CTkLabel(options_frame, text="DT0 Bounds:")
        self.dt0_bounds_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.dt0_bounds_frame = ctk.CTkFrame(options_frame)
        self.dt0_bounds_frame.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        
        self.dt0_lower = ctk.CTkEntry(self.dt0_bounds_frame, width=100, placeholder_text="Lower")
        self.dt0_lower.pack(side="left", padx=5)
        ctk.CTkLabel(self.dt0_bounds_frame, text="→").pack(side="left", padx=5)
        self.dt0_upper = ctk.CTkEntry(self.dt0_bounds_frame, width=100, placeholder_text="Upper")
        self.dt0_upper.pack(side="left", padx=5)

        # Number of starts
        row += 1
        ctk.CTkLabel(options_frame, text="Number of Starts:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.n_starts = ctk.CTkEntry(options_frame, width=100)
        self.n_starts.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        # Set default values
        self.d1_lower.insert(0, str(FVT_FITTING_DEFAULTS["D1_prime"]["lower_bound"]))
        self.d1_upper.insert(0, str(FVT_FITTING_DEFAULTS["D1_prime"]["upper_bound"]))
        self.d1_initial.insert(0, str(FVT_FITTING_DEFAULTS["D1_prime"]["initial"]))
        self.dt0_lower.insert(0, str(FVT_FITTING_DEFAULTS["DT0"]["lower_bound"]))
        self.dt0_upper.insert(0, str(FVT_FITTING_DEFAULTS["DT0"]["upper_bound"]))
        self.dt0_initial.insert(0, str(FVT_FITTING_DEFAULTS["DT0"]["initial"]))
        self.n_starts.insert(0, str(FVT_FITTING_DEFAULTS["n_starts"]))
        
        # Initial UI state
        self.update_fitting_inputs("D1' Only")

    def update_fitting_inputs(self, mode):
        """Update visible fitting inputs based on selected mode"""
        if mode == "D1' Only":
            # Hide DT0 controls
            self.dt0_initial_label.grid_remove()
            self.dt0_initial.grid_remove()
            self.dt0_bounds_label.grid_remove()
            self.dt0_bounds_frame.grid_remove()
        else:
            # Show DT0 controls
            self.dt0_initial_label.grid()
            self.dt0_initial.grid()
            self.dt0_bounds_label.grid()
            self.dt0_bounds_frame.grid()

    def create_section_frame(self, title):
        """Create a section frame with title"""
        section = ctk.CTkFrame(self.input_scroll)
        section.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(section, text=title,
                    font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        separator = ctk.CTkFrame(section, height=1)
        separator.pack(fill="x", padx=10)
        
        return section

    def create_parameter_bounds(self, parent, name, defaults):
        """Create parameter bounds and initial value inputs"""
        # Initial value
        init_row = ctk.CTkFrame(parent)
        init_row.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(init_row, text=f"{name} Initial:").pack(side="left", padx=5)
        
        initial = ctk.CTkEntry(init_row, width=120)
        initial.insert(0, str(defaults["initial"]))
        initial.pack(side="right", padx=5)
        
        # Bounds
        bounds_row = ctk.CTkFrame(parent)
        bounds_row.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(bounds_row, text=f"{name} Bounds:").pack(side="left", padx=5)
        
        bounds_entry = ctk.CTkFrame(bounds_row)
        bounds_entry.pack(side="right", padx=5)
        
        lower = ctk.CTkEntry(bounds_entry, width=80, placeholder_text="Lower")
        lower.insert(0, str(defaults["lower_bound"]))
        lower.pack(side="left", padx=2)
        
        ctk.CTkLabel(bounds_entry, text="→").pack(side="left", padx=2)
        
        upper = ctk.CTkEntry(bounds_entry, width=80, placeholder_text="Upper")
        upper.insert(0, str(defaults["upper_bound"]))
        upper.pack(side="left", padx=2)
        
        # Store references
        key = name.lower().replace(" ", "")
        self.bounds_entries[key] = {
            "lower": lower,
            "upper": upper,
            "initial": initial
        }

    def setup_results_content(self):
        """Set up fitting mode results tab"""
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

    def run_results(self):
        """Execute fitting workflow"""
        self.tabview.set("Results")  # Switch to results tab
        
        # Get parameters
        params = self.get_parameters()
        if not params:
            return
        
        # Get file path
        if self.file_source.get() == "preset":
            filename = self.file_selector.get()
            if not filename:
                self.show_error("Please select a data file")
                return
            data_path = os.path.join("data", "single_pressure", filename)
        else:
            data_path = self.file_entry.get()
            if not data_path:
                self.show_error("Please select a data file")
                return
            
        # Get stabilisation threshold
        try:
            stab_threshold = float(self.stab_threshold.get())
        except ValueError:
            self.show_error("Invalid stabilisation threshold value")
            return
            
        # Verify all required parameters
        required_model_params = ['thickness', 'diameter', 'flowrate']
        required_sim_params = ['dx']
        
        missing_model_params = [p for p in required_model_params if p not in params]
        missing_sim_params = [p for p in required_sim_params if p not in params['simulation']]
        
        if missing_model_params or missing_sim_params:
            error_msg = []
            if missing_model_params:
                error_msg.append(f"Missing model parameters: {', '.join(missing_model_params)}")
            if missing_sim_params:
                error_msg.append(f"Missing simulation parameters: {', '.join(missing_sim_params)}")
            self.show_error("\n".join(error_msg))
            return

        try:
            self.status_label.configure(text="Running fitting...")
            self.progress.set(0.2)
            
            # Get fitting mode
            mode = "both" if self.fit_option.get() == "D1' & DT0" else "D1"
            
            # Create fitting settings based on mode
            if mode == "D1":
                initial_guess = float(self.d1_initial.get())
                bounds = (float(self.d1_lower.get()), float(self.d1_upper.get()))
            else:
                initial_guess = (float(self.d1_initial.get()), float(self.dt0_initial.get()))
                bounds = ((float(self.d1_lower.get()), float(self.d1_upper.get())),
                         (float(self.dt0_lower.get()), float(self.dt0_upper.get())))
            
            fitting_settings = {
                'mode': mode,
                'initial_guess': initial_guess,
                'bounds': bounds,
                'n_starts': int(self.n_starts.get())
            }
            
            # Run fitting workflow
            model, fit_results, figures, Dprime_df, flux_df, exp_data = data_fitting_workflow(
                data_path=data_path,
                pressure=params.get('pressure', 1.0),
                temperature=params.get('temperature', 25.0),
                thickness=params['thickness'],
                diameter=params['diameter'],
                flowrate=params['flowrate'],
                DT_0=float(self.dt0_initial.get()),
                D1_prime=float(self.d1_initial.get()),
                fitting_settings=fitting_settings,
                stabilisation_threshold=stab_threshold,
                output_settings={
                    'display_plots': False,
                    'save_plots': False,
                    'save_data': False
                }
            )
            
            self.progress.set(0.6)
            
            # Clear the entire figure and create new subplots
            self.fig.clear()
            self.axes = {
                'diffusivity_profile': self.fig.add_subplot(221),
                'diffusivity_location': self.fig.add_subplot(222),
                'flux_time': self.fig.add_subplot(223),
                'flux_tau': self.fig.add_subplot(224)
            }
            
            # Display results text
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", "Fitting Results:\n")
            self.results_text.insert("end", f"D1 Prime: {fit_results['D1_prime']:.4e}\n")
            if 'DT0' in fit_results:
                self.results_text.insert("end", f"DT0: {fit_results['DT0']:.4e}\n")
            self.results_text.insert("end", f"RMSE: {fit_results['rmse']:.4e}\n")
            
            self.progress.set(0.8)
            
            # Plot results in the four subplots with explicit figure reference
            plot_diffusivity_profile(
                diffusivity_profile=Dprime_df,
                ax=self.axes['diffusivity_profile'],
                fig=self.fig,
                display=False
            )
            
            plot_diffusivity_location_profile(
                diffusivity_profile=Dprime_df,
                L=params['thickness'],
                T=flux_df['time'].max(),
                ax=self.axes['diffusivity_location'],
                fig=self.fig,
                display=False
            )
            
            plot_norm_flux_over_time(
                flux_data=flux_df,
                experimental_data=exp_data,
                ax=self.axes['flux_time'],
                fig=self.fig,
                display=False
            )
            
            plot_norm_flux_over_tau(
                flux_data=flux_df,
                experimental_data=exp_data,
                ax=self.axes['flux_tau'],
                fig=self.fig,
                display=False
            )
            
            # Update plot layout and display
            self.fig.tight_layout()
            self.canvas.draw()
            
            # Final status update
            self.status_label.configure(text="Complete")
            self.progress.set(1.0)
            
        except Exception as e:
            self.show_error(f"Fitting error: {str(e)}")
            self.status_label.configure(text="Error")
            self.progress.set(0)

    def get_parameters(self):
        """Get parameters from entries"""
        params = {}
        # Get model parameters
        for name, entry in self.parameter_entries.items():
            value = entry.get()
            if value:  # Only include non-empty values
                try:
                    params[name] = float(value)
                except ValueError:
                    self.show_error(f"Invalid value for {name}")
                    return None
        
        # Add default simulation parameter
        params['simulation'] = {'dx': 0.005}  # Use fixed dx value
        
        return params

    def show_error(self, message):
        """Show error message"""
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", f"Error: {message}")

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

    def create_preset_frame(self, parent):
        """Create frame for preset file selection"""
        preset_row = ctk.CTkFrame(parent)
        
        ctk.CTkLabel(preset_row, text="Preset File:").pack(side="left", padx=5)
        
        control_frame = ctk.CTkFrame(preset_row)
        control_frame.pack(side="right", padx=5)
        
        self.file_selector = ctk.CTkComboBox(
            control_frame,
            values=self.get_data_files(),
            width=200,
            state="readonly"
        )
        self.file_selector.pack(side="left", padx=2)
        
        ctk.CTkButton(control_frame, text="↻", width=30,
                     command=self.refresh_files).pack(side="left", padx=2)
        
        return preset_row

    def create_browse_frame(self, parent):
        """Create frame for file browser"""
        browse_row = ctk.CTkFrame(parent)
        
        ctk.CTkLabel(browse_row, text="File Path:").pack(side="left", padx=5)
        
        control_frame = ctk.CTkFrame(browse_row)
        control_frame.pack(side="right", padx=5)
        
        self.file_entry = ctk.CTkEntry(control_frame, width=200)
        self.file_entry.pack(side="left", padx=2)
        
        ctk.CTkButton(control_frame, text="Browse",
                     command=self.browse_file).pack(side="left", padx=2)
        
        return browse_row

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

    def on_file_selected(self, file_name):
        """Update thickness, diameter and flowrate based on selected file"""
        if not file_name:  # Skip if no file selected
            return
            
        print(f"Selected file: {file_name}")  # Debug print
            
        # Remove file extension for lookup
        base_name = os.path.splitext(file_name)[0]
        print(f"Base name: {base_name}")  # Debug print
        
        try:
            # Update thickness if available
            if base_name in THICKNESS_DICT:
                thickness = THICKNESS_DICT[base_name]
                print(f"Setting thickness to: {thickness}")  # Debug print
                self.parameter_entries["thickness"].delete(0, "end")
                self.parameter_entries["thickness"].insert(0, str(thickness))
            
            # Update flowrate if available
            if base_name in FLOWRATE_DICT:
                flowrate = FLOWRATE_DICT[base_name]
                print(f"Setting flowrate to: {flowrate}")  # Debug print
                self.parameter_entries["flowrate"].delete(0, "end")
                self.parameter_entries["flowrate"].insert(0, str(flowrate))
                
            # Default diameter remains constant
            self.parameter_entries["diameter"].delete(0, "end")
            self.parameter_entries["diameter"].insert(0, str(DEFAULTS["diameter"]))
            
            self.update_idletasks()  # Force GUI update
            
        except Exception as e:
            print(f"Error updating parameters: {e}")  # Debug print

class VariableFVTPlugin(ModelPlugin):
    def __init__(self, parent):
        super().__init__(parent)
    
    def setup_frames(self):
        """Create manual and fitting frames"""
        self.manual_frame = VariableFVTManual(self.parent)
        self.fitting_frame = VariableFVTFitting(self.parent)  # Fixed the typo 'fittime' to 'fitting_frame'
