# GUI Design Documentation

## Architecture Overview

The GUI follows a plugin-based architecture with clear separation of concerns:

```
PermeationAnalysisApp (main_window.py)
├── Main Window
│   ├── Sidebar
│   │   ├── Model Selection
│   │   └── Settings
│   └── Main Content Area
        └── Model Plugins
            ├── Constant Diffusivity Plugin
            │   ├── Manual Mode Frame
            │   └── Fitting Mode Frame
            └── Variable FVT Plugin
                ├── Manual Mode Frame
                └── Fitting Mode Frame
```

## Core Components

### 1. Main Application (`main_window.py`)
- Creates the root window and main layout
- Manages the sidebar with model selection and settings
- Loads and coordinates model plugins
- Handles global settings (UI scaling, theme)

### 2. Plugin System (`plugins/__init__.py`)
- Defines abstract base classes for plugins and frames
- `ModelPlugin`: Base class for model implementations
- `ModeFrame`: Base class for mode-specific frames (Manual/Fitting)
- `ScrollableFrame`: Utility class for scrollable content

### 3. Model Plugins
Each model is implemented as a plugin with two modes:
- Manual mode: Direct calculation with user inputs
- Fitting mode: Parameter fitting using experimental data

## Design Philosophy

### 1. Single Responsibility
- Each class has a specific responsibility
- Clear separation between UI components and business logic
- Model calculations are delegated to separate workflow modules

### 2. Frame Management
- Each model-mode combination (e.g., FVT-Manual, FVT-Fitting) is a separate frame
- Only one frame is visible at a time
- Frames are created at startup but hidden (`pack_forget`)
- Switching between frames is handled by the plugin manager

### 3. Tab-based Interface
Each mode frame contains two tabs:
1. Input Tab
   - Collects user inputs via forms
   - Input validation
   - Triggers calculations/fitting
   - Scrollable for extensive inputs

2. Results Tab
   - Displays calculation results
   - Shows plots and data visualization
   - Updates dynamically when new results are available

### 4. Workflow Integration
1. User Input Collection
   - Forms gather necessary parameters
   - Input validation before processing
   - Default values and presets available

2. Processing
   - Input parameters fed into appropriate workflow
   - Progress tracking during calculations
   - Error handling and user feedback

3. Results Display
   - Automatic switch to results tab
   - Multiple plot views
   - Text summary of results
   - Error messages if calculation fails

## Implementation Details

### 1. Frame Lifecycle
```
1. Plugin Initialization
   ├── Create Manual Frame (hidden)
   └── Create Fitting Frame (hidden)

2. Mode Selection
   ├── Hide current frame (pack_forget)
   └── Show selected frame (pack)

3. Tab Management
   ├── Input Tab (ScrollableFrame)
   └── Results Tab (Regular Frame)
```

### 2. Data Flow
```
User Input → Validation → Workflow Processing → Results Display
└── Input Tab                                  └── Results Tab
```

### 3. State Management
- Each frame maintains its own state
- Plugin manages active frame state
- Main window coordinates between plugins

## Future Considerations

1. Extensibility
   - Easy addition of new model plugins
   - Consistent interface for all models
   - Reusable components

2. Performance
   - Lazy loading of frames
   - Efficient plot updates
   - Background processing for long calculations

3. User Experience
   - Consistent layout across models
   - Intuitive workflow
   - Clear feedback and error messages
