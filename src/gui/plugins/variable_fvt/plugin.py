from .. import ModelPlugin
from .manual import VariableFVTManual
from .fitting import VariableFVTFitting

class VariableFVTPlugin(ModelPlugin):
    def __init__(self, parent):
        super().__init__(parent)
    
    def setup_frames(self):
        """Create manual and fitting frames"""
        self.manual_frame = VariableFVTManual(self.parent)
        self.fitting_frame = VariableFVTFitting(self.parent)
