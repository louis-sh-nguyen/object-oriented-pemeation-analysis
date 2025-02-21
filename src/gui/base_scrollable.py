import customtkinter as ctk

class ScrollableFrame(ctk.CTkScrollableFrame):
    """Base scrollable frame for content"""
    
    def __init__(self, container, **kwargs):
        super().__init__(container, **kwargs)
