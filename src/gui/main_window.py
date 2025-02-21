import customtkinter as ctk
from tkinter import Tk
from typing import Dict, Optional

class PermeationAnalysisApp:
    """Main application window"""
    
    def __init__(self):
        # Main Window setup
        self.root = Tk()
        self.root.title("Permeation Analysis Tool")
        self.root.geometry("1200x700")
        
        # Setup escape key binding
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        # Initialize UI components
        self.sidebar: Optional[ctk.CTkFrame] = None
        self.main_frame: Optional[ctk.CTkFrame] = None
        self.model_selector: Optional[ctk.CTkComboBox] = None
        self.mode_switch: Optional[ctk.CTkSwitch] = None
        self.theme_switch: Optional[ctk.CTkSwitch] = None
        
        # Create UI
        self.create_sidebar()
        self.create_main_content()
        
        # Initialize plugins
        self.current_model: Optional[str] = None
        self.plugins: Dict = {}
        self.load_plugins()
        
        # Show default view
        self.show_model("Constant Diffusivity")
    
    def create_sidebar(self):
        """Create sidebar with controls"""
        self.sidebar = ctk.CTkFrame(self.root, width=200, corner_radius=0)
        self.sidebar.pack_propagate(False)
        self.sidebar.pack(side="left", fill="y")
        
        # Title
        ctk.CTkLabel(self.sidebar, text="Permeation",
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20,0))
        ctk.CTkLabel(self.sidebar, text="Analyser",
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(0,20))
        
        # Controls
        self.create_model_frame()
        self.create_settings_frame()
    
    def create_model_frame(self):
        """Create model selection controls"""
        frame = ctk.CTkFrame(self.sidebar)
        frame.pack(fill="x", padx=10, pady=5)
        
        # Model selector
        ctk.CTkLabel(frame, text="Model Selection:",
                    font=ctk.CTkFont(weight="bold")).pack(pady=5)
        
        self.model_selector = ctk.CTkComboBox(
            frame,
            values=["Constant Diffusivity", "Variable FVT"],
            command=self.handle_model_selection,
            state="readonly"
        )
        self.model_selector.pack(fill="x", padx=5, pady=5)
        
        # Mode switch
        self.mode_switch = ctk.CTkSwitch(
            frame,
            text="Fitting Mode",
            command=self.toggle_mode
        )
        self.mode_switch.pack(pady=10)
    
    def create_settings_frame(self):
        """Create settings controls"""
        frame = ctk.CTkFrame(self.sidebar)
        frame.pack(side="bottom", fill="x", pady=20, padx=5)
        
        # UI scaling
        self.create_scaling_control(frame)
        
        # Theme toggle
        self.create_theme_control(frame)
    
    def create_scaling_control(self, parent):
        """Create UI scaling controls"""
        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row, text="UI Scaling:").pack(side="left", padx=5)
        scaling = ctk.CTkComboBox(
            row,
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.update_scaling,
            state="readonly",
            width=100
        )
        scaling.set("100%")
        scaling.pack(side="right", padx=5)
    
    def create_theme_control(self, parent):
        """Create theme controls"""
        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=5)
        
        ctk.CTkLabel(row, text="Theme:").pack(side="left", padx=5)
        self.theme_switch = ctk.CTkSwitch(
            row,
            text="Dark Mode",
            command=self.toggle_theme
        )
        self.theme_switch.pack(side="right", padx=5)
        self.theme_switch.select()
    
    def create_main_content(self):
        """Create main content area"""
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(side="right", fill="both", expand=True)
    
    def load_plugins(self):
        """Load model plugins"""
        # Import plugins using relative imports
        from .plugins.constant_diffusivity.plugin import ConstantDiffusivityPlugin
        from .plugins.variable_fvt.plugin import VariableFVTPlugin
        
        self.plugins = {
            "Constant Diffusivity": ConstantDiffusivityPlugin(self.main_frame),
            "Variable FVT": VariableFVTPlugin(self.main_frame)
        }
    
    def handle_model_selection(self, model_name: str):
        """Handle model selection"""
        self.show_model(model_name)
    
    def toggle_mode(self):
        """Toggle between Manual and Fitting modes"""
        if self.current_model:
            mode = "Fitting" if self.mode_switch.get() else "Manual"
            self.plugins[self.current_model].show_mode(mode)
    
    def show_model(self, model_name: str):
        """Switch to selected model"""
        if self.current_model != model_name:
            # Hide current frames
            for plugin in self.plugins.values():
                if plugin.current_frame:
                    plugin.current_frame.hide()

            # Switch to new model
            self.current_model = model_name
            if model_name in self.plugins:
                mode = "Fitting" if self.mode_switch.get() else "Manual"
                self.plugins[model_name].show_mode(mode)
    
    def update_scaling(self, value: str):
        """Update UI scaling"""
        scaling = int(value.replace('%', '')) / 100
        ctk.set_widget_scaling(scaling)
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        ctk.set_appearance_mode("Dark" if self.theme_switch.get() else "Light")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()
