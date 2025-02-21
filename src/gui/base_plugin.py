from abc import ABC, abstractmethod
from typing import Optional
from .base_frame import ModeFrame

class ModelPlugin(ABC):
    """Abstract base class for model plugins"""
    
    def __init__(self, parent):
        self.parent = parent
        self.manual_frame: Optional[ModeFrame] = None
        self.fitting_frame: Optional[ModeFrame] = None
        self.current_frame: Optional[ModeFrame] = None
        
        # Create frames
        self.setup_frames()
    
    @abstractmethod
    def setup_frames(self):
        """Create manual and fitting frames"""
        pass
    
    def show_mode(self, mode: str):
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
