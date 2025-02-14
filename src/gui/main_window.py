import customtkinter as ctk
from tkinter import Tk
from .plugins.constant_diffusivity_plugin import ConstantDiffusivityPlugin
from .plugins.variable_fvt_plugin import VariableFVTPlugin

class PermeationAnalysisApp:
    def __init__(self):
        # Main Window
        self.root = Tk()
        self.root.title("Permeation Analysis Tool")
        self.root.geometry("1200x700")

        # Initialize plugins
        self.plugins = {
            "Constant Diffusivity": ConstantDiffusivityPlugin(),
            "Variable FVT": VariableFVTPlugin()
        }
        
        # Create UI
        self._create_sidebar()
        self._create_main_content()
        
        # Show default view
        self.current_model = "Constant Diffusivity"
        self.show_model(self.current_model)

    def _create_sidebar(self):
        # Create sidebar
        self.sidebar = ctk.CTkFrame(self.root, width=250, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        
        # Title
        ctk.CTkLabel(self.sidebar, text="Permeation", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20,0))
        ctk.CTkLabel(self.sidebar, text="Analyser", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(0,20))
        
        # Model selection
        self._create_model_frame()
        
        # Bottom section
        self._create_bottom_frame()

    def _create_model_frame(self):
        # Model Selection Frame
        self.model_frame = ctk.CTkFrame(self.sidebar)
        self.model_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(self.model_frame, text="Model Selection:", 
                    font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        self.model_selector = ctk.CTkComboBox(
            self.model_frame,
            values=["Constant Diffusivity", "Variable FVT"],
            command=self.handle_model_selection,
            state="readonly"
        )
        self.model_selector.pack(fill="x", padx=5, pady=5)
        self.model_selector.set("Constant Diffusivity")
        
        self.mode_switch = ctk.CTkSwitch(
            self.model_frame, 
            text="Fitting Mode",
            command=self.toggle_mode
        )
        self.mode_switch.pack(pady=10)

    def _create_bottom_frame(self):
        self.bottom_frame = ctk.CTkFrame(self.sidebar)
        self.bottom_frame.pack(side="bottom", fill="x", pady=20)
        
        ctk.CTkLabel(self.bottom_frame, text="UI Scaling:").pack(pady=5)
        self.ui_scaling = ctk.CTkComboBox(
            self.bottom_frame, 
            values=["80%", "90%", "100%", "110%", "120%"],
            state="readonly",
            command=self.update_scaling
        )
        self.ui_scaling.set("100%")
        self.ui_scaling.pack(pady=5)
        
        ctk.CTkLabel(self.bottom_frame, text="Theme:").pack(pady=5)
        self.theme_switch = ctk.CTkSwitch(
            self.bottom_frame,
            text="Dark Mode",
            command=self.toggle_theme
        )
        self.theme_switch.select()
        self.theme_switch.pack(pady=5)

    def _create_main_content(self):
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(side="right", fill="both", expand=True)
        
        self.current_model = "Constant Diffusivity"
        
        self.constant_diff_frame = ctk.CTkFrame(self.main_frame)
        self.variable_fvt_frame = ctk.CTkFrame(self.main_frame)
        
        self.model_states = {
            "Constant Diffusivity": {"mode": "Manual"},
            "Variable FVT": {"mode": "Manual"}
        }
        
        self.add_content(self.constant_diff_frame, "Constant Diffusivity")
        self.add_content(self.variable_fvt_frame, "Variable FVT")
        self.show_model("Constant Diffusivity")

    def add_content(self, frame, model_name):
        """Add content for a model using its plugin"""
        # Get the plugin for this model
        plugin = self.plugins[model_name]
        
        # Create frames using plugin
        manual_frame = plugin.create_manual_frame(frame)
        fitting_frame = plugin.create_fitting_frame(frame) 
        
        # Store Frames
        frame.frames = {"Manual": manual_frame, "Fitting": fitting_frame}
        
        # Initial Mode
        self.switch_mode(frame, model_name, "Manual")

    def handle_model_selection(self, model_name):
        """Handle model selection from dropdown"""
        self.current_model = model_name
        self.show_model(model_name)
        
        # Update mode switch to match model's current mode
        current_mode = self.model_states[model_name]["mode"]
        self.mode_switch.select() if current_mode == "Fitting" else self.mode_switch.deselect()

    def show_model(self, model_name):
        """Switch displayed model content"""
        # Hide all frames
        self.constant_diff_frame.pack_forget()
        self.variable_fvt_frame.pack_forget()
        
        # Show selected model
        if (model_name == "Constant Diffusivity"):
            self.constant_diff_frame.pack(fill="both", expand=True)
        else:
            self.variable_fvt_frame.pack(fill="both", expand=True)

    def toggle_mode(self):
        """Toggle between Manual and Fitting modes"""
        if not self.current_model:
            return
            
        new_mode = "Fitting" if self.mode_switch.get() else "Manual"
        self.model_states[self.current_model]["mode"] = new_mode
        
        # Get corresponding frame
        frame = self.constant_diff_frame if self.current_model == "Constant Diffusivity" else self.variable_fvt_frame
        self.switch_mode(frame, self.current_model, new_mode)

    def switch_mode(self, frame, model_name, mode):
        """Switch between Manual and Fitting modes"""
        if hasattr(frame, 'current_frame'):
            frame.current_frame.pack_forget()
        frame.frames[mode].pack(fill="both", expand=True)
        frame.current_frame = frame.frames[mode]
        print(f"Switched to {mode} mode for {model_name}")
    
    def update_scaling(self, value):
        """Update UI scaling"""
        scaling = int(value.replace('%', '')) / 100
        ctk.set_widget_scaling(scaling)
        
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        ctk.set_appearance_mode("Dark" if self.theme_switch.get() else "Light")
    
    def run(self):
        self.root.mainloop()