from abc import ABC, abstractmethod
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

__all__ = ['ModelPlugin', 'ScrollableFrame', 'ModeFrame']

class ScrollableFrame(ctk.CTkScrollableFrame):
    """Base scrollable frame for all content"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

class ModeFrame(ctk.CTkFrame):
    """Base class for mode-specific frames (Manual/Fitting)"""
    def __init__(self, parent, mode_name, model_name):
        super().__init__(parent)
        self.mode_name = mode_name
        self.model_name = model_name
        
        # Add title at the top of the frame (before tabview)
        ctk.CTkLabel(self, 
                    text=f"{model_name} - {mode_name} Mode",
                    font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20,10))
        
        # Create tabview
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=(0,10))  # Adjusted top padding
        
        # Create tabs
        self.input_tab = self.tabview.add("Input")
        self.results_tab = self.tabview.add("Results")
        
        # Create scrollable container for input only
        self.input_scroll = ScrollableFrame(self.input_tab)
        self.input_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create regular frame for results
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
    
    def show(self):
        """Show this mode frame"""
        self.pack(fill="both", expand=True)
    
    def hide(self):
        """Hide this mode frame"""
        self.pack_forget()

class ModelPlugin(ABC):
    """Base class for model plugins"""
    
    def __init__(self, parent):
        self.parent = parent
        self.manual_frame = None
        self.fitting_frame = None
        self.current_frame = None
        
        # Create frames
        self.setup_frames()
    
    @abstractmethod
    def setup_frames(self):
        """Create manual and fitting frames"""
        pass
    
    def show_mode(self, mode):
        """Switch to specified mode"""
        if self.current_frame:
            self.current_frame.hide()
            
        if mode == "Manual":
            if self.manual_frame:
                self.manual_frame.show()
                self.current_frame = self.manual_frame
        else:  # Fitting mode
            if self.fitting_frame:
                self.fitting_frame.show()
                self.current_frame = self.fitting_frame
