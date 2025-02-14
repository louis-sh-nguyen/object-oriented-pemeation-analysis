from abc import ABC, abstractmethod
import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ScrollableFrame(ctk.CTkScrollableFrame):
    """Base scrollable frame for all content"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

class ModelPlugin(ABC):
    """Base class for model plugins"""
    
    def create_base_frame(self, parent, mode_name, model_name):
        """Create standard frame with Input/Results tabs"""
        frame = ctk.CTkFrame(parent)
        
        # Create tabview for Input and Results
        tabview = ctk.CTkTabview(frame)
        tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        input_tab = tabview.add("Input")
        results_tab = tabview.add("Results")
        
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
        # Results text area
        self.results_text = ctk.CTkTextbox(parent, height=100)
        self.results_text.pack(fill="x", pady=5, padx=10)
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(parent)
        self.progress.pack(fill="x", pady=5, padx=10)
        self.progress.set(0)
        
        # Plot area
        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=5)

    @abstractmethod
    def create_input_content(self, parent, mode):
        """Create input content specific to each model"""
        pass

    @abstractmethod
    def create_manual_frame(self, parent):
        """Create the manual mode frame"""
        pass

    @abstractmethod
    def create_fitting_frame(self, parent):
        """Create the fitting mode frame"""
        pass

    @abstractmethod
    def run_fitting(self):
        """Execute the fitting workflow"""
        pass

    @abstractmethod
    def generate_manual_results(self):
        """Generate results in manual mode"""
        pass
