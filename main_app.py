import customtkinter as ctk
from src.gui.main_window import PermeationAnalysisApp

def main():
    # Initialize CustomTkinter settings
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    
    # Create and run application
    app = PermeationAnalysisApp()
    
    # Display hint about escape key
    print("Press ESC to exit the application")
    
    app.run()

if __name__ == "__main__":
    main()