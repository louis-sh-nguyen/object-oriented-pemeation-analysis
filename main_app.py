import customtkinter as ctk
from src.gui.main_window import PermeationAnalysisApp

def main():
    # Initialize CustomTkinter settings
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    # Create and run application
    app = PermeationAnalysisApp()
    
    # Select Variable FVT model and Manual mode by default for testing
    app.model_selector.set("Variable FVT")
    app.handle_model_selection("Variable FVT")
    app.mode_switch.deselect()  # This will set it to Manual mode
    
    # Display hint about escape key
    print("Press ESC to exit the application")
    
    app.run()

if __name__ == "__main__":
    main()