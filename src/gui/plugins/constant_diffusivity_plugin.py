import os
import customtkinter as ctk
from . import ModelPlugin

class ConstantDiffusivityPlugin(ModelPlugin):
    def __init__(self):
        self.parameter_entries = {}
        self.data_dir = os.path.join("data", "single_pressure")

    def create_manual_frame(self, parent):
        return self.create_base_frame(parent, "Manual", "Constant Diffusivity")

    def create_fitting_frame(self, parent):
        return self.create_base_frame(parent, "Fitting", "Constant Diffusivity")

    def create_input_content(self, parent, mode):
        """Create input content specific to Constant Diffusivity model"""
        # Create parameter frame
        self.create_parameter_frame(parent)
        
        if mode == "Fitting":
            # Add file selection and fitting options
            self.create_file_selection(parent)
            self.create_fitting_options(parent)
            # Fitting button
            ctk.CTkButton(parent, text="Start Fitting", 
                         command=self.generate_fitting_results).pack(pady=10)
        else:
            # Manual mode button
            ctk.CTkButton(parent, text="Generate Results", 
                         command=self.generate_manual_results).pack(pady=10)

    def create_parameter_frame(self, parent):
        """Create a frame containing all parameter inputs"""
        param_frame = ctk.CTkFrame(parent)
        param_frame.pack(fill="x", padx=10, pady=5)
        
        # Parameter definitions
        parameters = [
            ("Required Parameters", [
                ("thickness", "Thickness (mm)", True),
                ("diameter", "Diameter (cm)", True),
                ("flowrate", "Flow Rate (cm³/min)", True),
            ]),
            ("Optional Parameters", [
                ("pressure", "Pressure (bar)", False),
                ("temperature", "Temperature (°C)", False),
                ("D0", "Diffusivity (m²/s)", False),
            ])
        ]
        
        # Create sections for required and optional parameters
        for section_title, params in parameters:
            # Section title
            ctk.CTkLabel(param_frame, 
                        text=section_title,
                        font=ctk.CTkFont(weight="bold")).pack(pady=5)
            
            # Parameters grid
            for param_name, param_label, required in params:
                param_row = ctk.CTkFrame(param_frame)
                param_row.pack(fill="x", padx=5, pady=2)
                
                # Label
                label_text = f"{param_label}{'*' if required else ''}"
                ctk.CTkLabel(param_row, text=label_text).pack(side="left", padx=5)
                
                # Entry
                entry = ctk.CTkEntry(param_row, width=120)
                entry.pack(side="right", padx=5)
                
                # Store entry reference
                self.parameter_entries[param_name] = entry
        
        return param_frame

    def create_file_selection(self, parent):
        """Create file selection section"""
        file_frame = ctk.CTkFrame(parent)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(file_frame, text="Data File:").pack(side="left", padx=5)
        self.file_entry = ctk.CTkEntry(file_frame, width=250)
        self.file_entry.pack(side="left", padx=5)
        ctk.CTkButton(file_frame, text="Browse", 
                     command=self.browse_file).pack(side="left", padx=5)

    def create_fitting_options(self, parent):
        """Create fitting options section"""
        options_frame = ctk.CTkFrame(parent)
        options_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(options_frame, text="Fit:", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.fit_option = ctk.CTkComboBox(
            options_frame,
            values=["D0"],
            state="readonly"
        )
        self.fit_option.pack(pady=5)

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

    def generate_fitting_results(self):
        """Execute the fitting workflow"""
        params = self.get_parameters()
        if not params:
            return
            
        # Get selected file path
        data_path = self.file_entry.get()
        if not data_path:
            self.show_error("Please select a data file")
            return
            
        if not os.path.exists(data_path):
            self.show_error("Selected data file not found")
            return
            
        # TODO: Implement actual fitting logic
        # from models.constant_diffusivity import fit_data
        # results = fit_data(data_path, **params)
        pass

    def generate_manual_results(self):
        """Generate results in manual mode"""
        params = self.get_parameters()
        if not params:
            return
            
        # TODO: Implement actual manual calculation logic
        # from models.constant_diffusivity import calculate_manual
        # results = calculate_manual(**params)
        pass