import os
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ...base_frame import ModeFrame
from ....utils.defaults import DEFAULTS, FVT_FITTING_DEFAULTS
from ....models.single_pressure.variable_diffusivity_fvt.workflow import manual_workflow
from ....models.single_pressure.variable_diffusivity_fvt.plotting import (
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
            self.results_text.insert("end", f"D1': {params['D1_prime']:.4e}\n")
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
