from abc import ABC, abstractmethod
import customtkinter as ctk
from .base_scrollable import ScrollableFrame

class ModeFrame(ctk.CTkFrame, ABC):
    """Abstract base class for mode-specific frames (Manual/Fitting)"""
    
    def __init__(self, parent, mode_name: str, model_name: str):
        super().__init__(parent)
        self.mode_name = mode_name
        self.model_name = model_name
        
        # Add title
        ctk.CTkLabel(self, 
                    text=f"{model_name} - {mode_name} Mode",
                    font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20,10))
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(0,10))
        
        # Create tabs
        self.input_tab = self.tabview.add("Input")
        self.results_tab = self.tabview.add("Results")
        
        # Create scrollable input container
        self.input_scroll = ScrollableFrame(self.input_tab)
        self.input_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create results frame
        self.results_frame = ctk.CTkFrame(self.results_tab)
        self.results_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Initialize UI elements
        self.setup_input_content()
        self.setup_results_content()
    
    @abstractmethod
    def setup_input_content(self):
        """Set up the input tab content"""
        pass
    
    @abstractmethod
    def setup_results_content(self):
        """Set up the results tab content"""
        pass

    @abstractmethod
    def run_results(self):
        """Run calculations or fitting process"""
        pass
    
    def show_error(self, message: str):
        """Show error message in results text area"""
        if hasattr(self, 'results_text'):
            self.results_text.delete("1.0", "end")
            self.results_text.insert("1.0", f"Error: {message}")
    
    def show(self):
        """Show this mode frame"""
        self.pack(fill="both", expand=True)
    
    def hide(self):
        """Hide this mode frame"""
        self.pack_forget()
