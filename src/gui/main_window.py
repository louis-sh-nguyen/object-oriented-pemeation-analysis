import customtkinter as ctk
from tkinter import Tk

class PermeationAnalysisApp:
    def __init__(self):
        # Main Window
        self.root = Tk()
        self.root.title("Permeation Analysis Tool")
        self.root.geometry("1200x700")
        
        # Setup escape key binding
        self.root.bind('<Escape>', lambda e: self.root.quit())
        
        # Create main UI elements
        self.create_sidebar()
        self.create_main_content()
        
        # Initialize state
        self.current_model = None
        self.plugins = {}
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
        
        # Model selection
        self.create_model_frame()
        
        # Settings
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
        
        # UI scaling row
        scaling_row = ctk.CTkFrame(frame)
        scaling_row.pack(fill="x", pady=5)
        ctk.CTkLabel(scaling_row, text="UI Scaling:").pack(side="left", padx=5)
        scaling = ctk.CTkComboBox(
            scaling_row,
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.update_scaling,
            state="readonly",
            width=100
        )
        scaling.set("100%")
        scaling.pack(side="right", padx=5)
        
        # Theme toggle row
        theme_row = ctk.CTkFrame(frame)
        theme_row.pack(fill="x", pady=5)
        ctk.CTkLabel(theme_row, text="Theme:").pack(side="left", padx=5)
        self.theme_switch = ctk.CTkSwitch(
            theme_row,
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
        from .plugins.constant_diffusivity_plugin import ConstantDiffusivityPlugin
        from .plugins.variable_fvt_plugin import VariableFVTPlugin
        
        self.plugins = {
            "Constant Diffusivity": ConstantDiffusivityPlugin(self.main_frame),
            "Variable FVT": VariableFVTPlugin(self.main_frame)
        }
    
    # Event handlers
    def handle_model_selection(self, model_name):
        """Handle model selection"""
        self.show_model(model_name)
    
    def toggle_mode(self):
        """Toggle between Manual and Fitting modes"""
        if self.current_model:
            mode = "Fitting" if self.mode_switch.get() else "Manual"
            self.plugins[self.current_model].show_mode(mode)
    
    def show_model(self, model_name):
        """Switch to selected model"""
        # Only switch if selecting a different model
        if self.current_model != model_name:
            # Hide all current frames first
            for plugin in self.plugins.values():
                if plugin.current_frame:
                    plugin.current_frame.hide()

            # Switch to new model
            self.current_model = model_name
            if model_name in self.plugins:
                mode = "Fitting" if self.mode_switch.get() else "Manual"
                self.plugins[model_name].show_mode(mode)
    
    def update_scaling(self, value):
        """Update UI scaling"""
        scaling = int(value.replace('%', '')) / 100
        ctk.set_widget_scaling(scaling)
    
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        ctk.set_appearance_mode("Dark" if self.theme_switch.get() else "Light")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()
