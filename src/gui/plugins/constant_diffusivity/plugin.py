from ...base_plugin import ModelPlugin
from .manual import ConstantDiffusivityManual
from .fitting import ConstantDiffusivityFitting

class ConstantDiffusivityPlugin(ModelPlugin):
    def __init__(self, parent):
        super().__init__(parent)
    
    def setup_frames(self):
        """Create manual and fitting frames"""
        self.manual_frame = ConstantDiffusivityManual(self.parent)
        self.fitting_frame = ConstantDiffusivityFitting(self.parent)
