import os
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from . import ModelPlugin, ScrollableFrame  # Add ScrollableFrame to import
from src.utils.defaults import (
    DEFAULTS,
    THICKNESS_DICT,
    FLOWRATE_DICT,
    FVT_FITTING_DEFAULTS
)
from src.models.single_pressure.variable_diffusivity_fvt.workflow import data_fitting_workflow
from src.models.single_pressure.variable_diffusivity_fvt.plotting import plot_norm_flux_over_time

class VariableFVTPlugin(ModelPlugin):
    def __init__(self):
        self.parameter_entries = {}
        self.data_dir = os.path.join("data", "single_pressure")
        self.current_tabview = None
        self.root = None  # Add this line

    def get_data_files(self):
        """Get list of Excel files in data directory"""
        try:
            files = [f for f in os.listdir(self.data_dir) 
                    if f.endswith(('.xlsx', '.xls'))]
            return files
        except Exception as e:
            print(f"Error reading data directory: {e}")
            return []

    def create_parameter_frame(self, parent):
        """Create a frame containing all parameter inputs with default values"""
        param_frame = ctk.CTkFrame(parent)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Parameter definitions with default values
        parameters = [
            ("Required Parameters", [
                ("thickness", "Thickness (mm)", True, str(DEFAULTS["thickness"])),
                ("diameter", "Diameter (cm)", True, str(DEFAULTS["diameter"])),
                ("flowrate", "Flow Rate (cm³/min)", True, str(DEFAULTS["flowrate"])),
            ]), 
            ("Optional Parameters", [
                ("pressure", "Pressure (bar)", False, ""),
                ("temperature", "Temperature (°C)", False, ""),
                ("D1_prime", "D1 Prime", False, ""),
                ("DT0", "DT0", False, "")
            ])
        ]
        
        # Create sections for required and optional parameters
        for section_title, params in parameters:
            # Section title
            ctk.CTkLabel(param_frame, 
                        text=section_title,
                        font=ctk.CTkFont(weight="bold")).pack(pady=5)
            
            # Parameters grid
            for param_name, param_label, required, default_value in params:
                param_row = ctk.CTkFrame(param_frame)
                param_row.pack(fill="x", padx=5, pady=2)
                
                # Label
                label_text = f"{param_label}{'*' if required else ''}"
                ctk.CTkLabel(param_row, text=label_text).pack(side="left", padx=5)
                
                # Entry with default value
                entry = ctk.CTkEntry(param_row, width=120)
                if default_value:
                    entry.insert(0, default_value)
                entry.pack(side="right", padx=5)
                
                # Store entry reference
                self.parameter_entries[param_name] = entry
        
        return param_frame

    def create_manual_frame(self, parent):
        frame = self.create_base_frame(parent, "Manual", "Variable FVT")
        return frame

    def create_fitting_frame(self, parent):
        frame = self.create_base_frame(parent, "Fitting", "Variable FVT")
        return frame

    def create_input_content(self, parent, mode):
        """Create input content specific to Variable FVT model"""
        # Parameters Box for all inputs
        params_box = ctk.CTkFrame(parent)
        params_box.pack(fill="x", padx=10, pady=5)
        
        # Parameters Box Title
        ctk.CTkLabel(params_box, text="Model Parameters", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
        
        # Parameter frame
        self.create_parameter_frame(params_box)
        
        # Add file selection section if in fitting mode
        if (mode == "Fitting"):
            # Separator line
            separator = ctk.CTkFrame(params_box, height=2)
            separator.pack(fill="x", padx=20, pady=10)
            
            # Add file selection to same box
            self.create_file_selection(params_box)
            
            # Fitting Settings Title
            ctk.CTkLabel(parent, text="Fitting Settings",
                        font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
            
            # Fitting Mode Box
            fitting_box = ctk.CTkFrame(parent)
            fitting_box.pack(fill="x", padx=10, pady=5)
            
            # Add fitting mode and options
            self.create_fitting_options(fitting_box)
            
            # Fitting button
            ctk.CTkButton(parent, text="Start Fitting", 
                         command=self.run_fitting).pack(pady=10)
        else:
            # Manual mode button
            ctk.CTkButton(parent, text="Generate Results", 
                         command=self.generate_manual_results).pack(pady=10)

    def create_file_selection(self, parent):
        """Create file selection and stabilisation threshold frame"""
        file_frame = ctk.CTkFrame(parent)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        # Title for section
        ctk.CTkLabel(file_frame, text="Data Selection", 
                    font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
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
            width=200,
            state="readonly"
        )
        self.file_selector.pack(side="left", padx=5)
        ctk.CTkButton(self.preset_frame, text="↻", width=30,
                     command=self.refresh_files).pack(side="left", padx=5)
        
        # Browse file frame
        self.browse_frame = ctk.CTkFrame(self.file_select_container)
        self.file_entry = ctk.CTkEntry(self.browse_frame, width=250)
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

        # Add callback for file selection
        self.file_selector.configure(command=lambda x: self.on_file_selected(x))

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
        
        # Fit mode selector at the top
        row = 0
        ctk.CTkLabel(options_frame, text="Mode:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.fit_option = ctk.CTkComboBox(
            options_frame,
            values=["D1 Prime Only", "D1 Prime & DT0"],
            command=self.update_fitting_inputs,
            state="readonly",
            width=120
        )
        self.fit_option.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.fit_option.set("D1 Prime Only")

        # D1 Prime bounds
        row += 1
        ctk.CTkLabel(options_frame, text="D1 Prime Bounds:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        bounds_frame = ctk.CTkFrame(options_frame)
        bounds_frame.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.d1_lower = ctk.CTkEntry(bounds_frame, width=100, placeholder_text="Lower")
        self.d1_lower.pack(side="left", padx=5)
        ctk.CTkLabel(bounds_frame, text="→").pack(side="left", padx=5)
        self.d1_upper = ctk.CTkEntry(bounds_frame, width=100, placeholder_text="Upper")
        self.d1_upper.pack(side="left", padx=5)
        
        # D1 Prime initial guess
        row += 1
        ctk.CTkLabel(options_frame, text="D1 Prime Initial:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.d1_initial = ctk.CTkEntry(options_frame, width=100)
        self.d1_initial.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        
        # DT0 bounds (initially hidden)
        row += 1
        self.dt0_bounds_label = ctk.CTkLabel(options_frame, text="DT0 Bounds:")
        self.dt0_bounds_frame = ctk.CTkFrame(options_frame)
        self.dt0_lower = ctk.CTkEntry(self.dt0_bounds_frame, width=100, placeholder_text="Lower")
        self.dt0_lower.pack(side="left", padx=5)
        ctk.CTkLabel(self.dt0_bounds_frame, text="→").pack(side="left", padx=5)
        self.dt0_upper = ctk.CTkEntry(self.dt0_bounds_frame, width=100, placeholder_text="Upper")
        self.dt0_upper.pack(side="left", padx=5)
        
        # DT0 initial guess (initially hidden)
        row += 1
        self.dt0_initial_label = ctk.CTkLabel(options_frame, text="DT0 Initial:")
        self.dt0_initial = ctk.CTkEntry(options_frame, width=100)
        
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
        self.update_fitting_inputs("D1 Prime Only")

    def update_fitting_inputs(self, mode):
        """Update visible fitting inputs based on selected mode"""
        if mode == "D1 Prime Only":
            self.dt0_bounds_label.grid_forget()
            self.dt0_bounds_frame.grid_forget()
            self.dt0_initial_label.grid_forget()
            self.dt0_initial.grid_forget()
        else:
            self.dt0_bounds_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
            self.dt0_bounds_frame.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
            self.dt0_initial_label.grid(row=4, column=0, sticky="w", padx=5, pady=5)
            self.dt0_initial.grid(row=4, column=1, sticky="w", padx=5, pady=5)

    def create_results_frame(self, parent):
        """Create a frame to display results"""
        results_frame = ctk.CTkFrame(parent)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Results text area
        self.results_text = ctk.CTkTextbox(results_frame, height=100)
        self.results_text.pack(fill="x", pady=5)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(results_frame)
        self.progress.pack(fill="x", pady=5)
        self.progress.set(0)
        
        # Plot area
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=5)
        
        return results_frame

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
        """Show error message in results text"""
        self.results_text.delete("0.0", "end")
        self.results_text.insert("0.0", f"Error: {message}")

    def browse_file(self):
        """Open file browser dialog"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, filename)

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

    def toggle_file_source(self):
        """Toggle between preset files and browse options"""
        if self.file_source.get() == "preset":
            self.browse_frame.pack_forget()
            self.preset_frame.pack(fill="x")
        else:
            self.preset_frame.pack_forget()
            self.browse_frame.pack(fill="x")

    def get_selected_file(self):
        """Get the currently selected file path"""
        if self.file_source.get() == "preset":
            selected_file = self.file_selector.get()
            if selected_file:
                return os.path.join(self.data_dir, selected_file)
        else:
            return self.file_entry.get()
        return None

    def get_fitting_params(self):
        """Get fitting parameters based on selected mode"""
        mode = self.fit_option.get()
        n_starts = int(self.n_starts.get())
        
        # Convert UI mode selection to model's mode values
        model_mode = 'D1' if mode == "D1 Prime Only" else 'both'
        
        params = {
            'n_starts': n_starts,
            'mode': model_mode  # Use the converted mode value
        }
        
        # Get bounds and initial point based on mode
        if model_mode == 'D1':  # Changed condition to match model's mode values
            params.update({
                'bounds': (float(self.d1_lower.get()), float(self.d1_upper.get())),
                'initial_guess': float(self.d1_initial.get())
            })
        else:  # mode == 'both'
            params.update({
                'bounds': (
                    (float(self.d1_lower.get()), float(self.d1_upper.get())),
                    (float(self.dt0_lower.get()), float(self.dt0_upper.get()))
                ),
                'initial_guess': (float(self.d1_initial.get()), float(self.dt0_initial.get()))
            })
        
        # Add stabilisation threshold
        try:
            stab_threshold = float(self.stab_threshold.get())
            params['stabilisation_threshold'] = stab_threshold
        except ValueError:
            params['stabilisation_threshold'] = 0.002  # Default value
            
        return params

    def run_fitting(self):
        """Execute the fitting workflow"""
        # Switch to Results tab immediately
        self.current_tabview.set("Results")
        
        # Clear previous results and show initial status
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "Starting fitting process...\n")
        self.fitting_status.configure(text="Initializing...")
        self.progress.set(0)
        self.ax.clear()
        self.canvas.draw()
        
        # Update UI
        self.root.update()

        # Get parameters and validate
        params = self.get_parameters()
        if not params:
            return
            
        # Get selected file path
        data_path = self.get_selected_file()
        if not data_path:
            self.show_error("Please select a data file")
            return
            
        if not os.path.exists(data_path):
            self.show_error("Selected data file not found")
            return
            
        # Get fitting parameters
        try:
            fitting_params = self.get_fitting_params()
        except ValueError as e:
            self.show_error(f"Invalid fitting parameter: {str(e)}")
            return
            
        try:
            # Run fitting workflow with progress updates
            self.fitting_status.configure(text="Fitting in progress...")
            self.progress.set(0.2)
            self.root.update()

            def progress_callback(iteration, total, best_params, best_rmse):
                """Callback to update fitting progress"""
                progress = (iteration + 1) / total
                self.progress.set(progress)
                
                # Update results text with current iteration info
                self.results_text.delete("1.0", "end")
                self.results_text.insert("1.0", f"Fitting Progress:\n")
                self.results_text.insert("end", f"Iteration: {iteration + 1}/{total}\n")
                self.results_text.insert("end", "\nBest Parameters:\n")
                for param, value in best_params.items():
                    self.results_text.insert("end", f"{param}: {value:.4e}\n")
                self.results_text.insert("end", f"\nCurrent RMSE: {best_rmse:.4e}\n")
                
                self.fitting_status.configure(text=f"Iteration {iteration + 1}/{total}")
                self.root.update()
            
            # Run workflow with callback
            model, fit_results, figures, Dprime_df, flux_df = data_fitting_workflow(
                data_path=data_path,
                pressure=params.get('pressure', None),
                temperature=params.get('temperature', None),
                thickness=params['thickness'],
                diameter=params['diameter'],
                flowrate=params['flowrate'],
                DT_0=params.get('DT0', 2.87e-7),
                D1_prime=params.get('D1_prime', 2.0),
                stabilisation_threshold=fitting_params['stabilisation_threshold'],
                fitting_settings={
                    'mode': fitting_params['mode'],
                    'initial_guess': fitting_params['initial_guess'],
                    'bounds': fitting_params['bounds'],
                    'n_starts': fitting_params['n_starts'],
                    'track_fitting_progress': True,  # Enable progress tracking
                    'progress_callback': progress_callback  # Add callback
                },
                output_settings={
                    'display_plots': False,
                    'save_plots': False,
                    'save_data': False
                }
            )
            
            # Update progress
            self.fitting_status.configure(text="Generating results...")
            self.progress.set(0.8)
            self.root.update()
            
            # Display results
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", "Fitting Results:\n")
            self.results_text.insert("end", f"D1 Prime: {fit_results['D1_prime']:.4e}\n")
            if 'DT_0' in fit_results:
                self.results_text.insert("end", f"DT0: {fit_results['DT_0']:.4e}\n")
            self.results_text.insert("end", f"RMSE: {fit_results['rmse']:.4e}\n")
            
            # Clear current axis and plot normalized flux vs time
            self.ax.clear()
            plot_norm_flux_over_time(
                flux_data=flux_df,
                ax=self.ax,
                display=False
            )
            
            # Update canvas
            self.canvas.draw()
            
            # Final progress update
            self.fitting_status.configure(text="Completed")
            self.progress.set(1.0)
            
        except Exception as e:
            self.show_error(f"Fitting error: {str(e)}")
            self.fitting_status.configure(text="Error")
            self.progress.set(0)

    def generate_manual_results(self):
        """Generate results in manual mode"""
        params = self.get_parameters()
        if not params:
            return
            
        # TODO: Implement actual manual calculation logic
        pass

    def on_file_selected(self, file_name):
        """Update thickness and flowrate based on selected file"""
        # Remove file extension for lookup
        base_name = os.path.splitext(file_name)[0]
        
        # Update thickness if available
        if (base_name in THICKNESS_DICT):
            self.parameter_entries["thickness"].delete(0, "end")
            self.parameter_entries["thickness"].insert(0, str(THICKNESS_DICT[base_name]))
        
        # Update flowrate if available
        if (base_name in FLOWRATE_DICT):
            self.parameter_entries["flowrate"].delete(0, "end")
            self.parameter_entries["flowrate"].insert(0, str(FLOWRATE_DICT[base_name]))

    def create_base_frame(self, parent, mode_name, model_name):
        """Create standard frame with Input/Results tabs"""
        frame = ctk.CTkFrame(parent)
        
        # Get reference to root window
        self.root = parent.winfo_toplevel()  # Add this line
        
        # Create tabview for Input and Results
        self.current_tabview = ctk.CTkTabview(frame)  # Store reference
        self.current_tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        input_tab = self.current_tabview.add("Input")
        results_tab = self.current_tabview.add("Results")
        
        # Create scrollable containers for each tab
        input_scroll = ScrollableFrame(input_tab)
        input_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        results_scroll = ScrollableFrame(results_tab)
        results_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Title in Input tab
        ctk.CTkLabel(input_scroll, 
                    text=f"{model_name} - {mode_name} Mode", 
                    font=ctk.CTkFont(size=20, weight="bold")).pack(pady=20)
        
        # Create content for each tab
        self.create_input_content(input_scroll, mode_name)
        self.create_results_content(results_scroll)
        
        return frame

    def create_results_content(self, parent):
        """Create standard results display"""
        # Progress frame
        progress_frame = ctk.CTkFrame(parent)
        progress_frame.pack(fill="x", pady=5, padx=10)
        
        # Fitting status
        self.fitting_status = ctk.CTkLabel(progress_frame, text="Ready")
        self.fitting_status.pack(side="left", padx=5)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(progress_frame)
        self.progress.pack(side="left", fill="x", expand=True, padx=5)
        self.progress.set(0)
        
        # Results text area
        self.results_text = ctk.CTkTextbox(parent, height=100)
        self.results_text.pack(fill="x", pady=5, padx=10)
        
        # Plot area
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=5)
