# GUI Design Documentation

## Project Structure
```
src/gui_new/
├── __init__.py
├── base_frame.py         # ModeFrame base class
├── base_plugin.py        # ModelPlugin base class
├── base_scrollable.py    # ScrollableFrame utility
├── main_window.py        # Main application window
└── plugins/
    ├── __init__.py      # Plugin system exports
    ├── constant_d/      # Constant diffusivity plugin
    │   ├── __init__.py
    │   ├── manual.py
    │   ├── fitting.py
    │   └── plugin.py
    └── variable_fvt/     # Variable FVT plugin
        ├── __init__.py
        ├── manual.py
        ├── fitting.py
        └── plugin.py
```

## Core Components

### Base Classes
1. ModeFrame (base_frame.py)`)
   - Base class for all mode frames (Manual/Fitting)
   - Handles tab management and layout
   - Abstract methods for content setup

2. ModelPlugin (base_plugin.py)`)
   - Base class for model plugins
   - Manages mode frames
   - Coordinates frame switching

3. ScrollableFrame (base_scrollable.py)`)
   - Utility class for scrollable content
   - Used in input tabs

### Plugin System
The plugin system uses composition over inheritance:

```python
ModelPlugin
├── Manual Frame (ModeFrame)
│   ├── Input Tab (ScrollableFrame)
│   └── Results Tab
└── Fitting Frame (ModeFrame)
    ├── Input Tab (ScrollableFrame)
    └── Results Tab
```

### Mode Frame Structure
Each mode frame (Manual/Fitting) follows this layout:

```
ModeFrame
├── Title
├── Input Tab
│   ├── Model Parameters
│   ├── Data Selection (Fitting only)
│   ├── Simulation Settings (Manual only)
│   └── Action Button
└── Results Tab
    ├── Progress Section
    ├── Results Text
    └── Plot Grid
```

## Workflows

### Manual Mode
1. Parameter Input
   - Required/Optional parameters
   - Simulation settings
   - Input validation

2. Calculation
   - Direct calculation with given parameters
   - Progress tracking
   - Error handling

3. Results Display
   - Multiple plot views
   - Parameter summary

### Fitting Mode
1. Data Selection
   - Preset file selection
   - File browser option
   - Auto-parameter population

2. Parameter Setup
   - Model parameters
   - Fitting settings
   - Initial values and bounds

3. Fitting Process
   - Progress tracking
   - RMSE calculation
   - Result visualization

## Implementation Details

### Frame Management
```python
class ModelPlugin:
    def setup_frames(self):
        self.manual_frame = ManualFrame(self.parent)
        self.fitting_frame = FittingFrame(self.parent)
        self.current_frame = None

    def show_mode(self, mode):
        if self.current_frame:
            self.current_frame.hide()
        
        frame = self.manual_frame if mode == "Manual" else self.fitting_frame
        frame.show()
        self.current_frame = frame
```

## Data Flow

### Component Interaction
```
Plugin Manager <──── Main Window ───> Model Selection
      │                                    │
      ├── Load Plugin ────────────────────>│
      │                                    │
      ├── Initialize Frames ──> Active Frame
      │                             │
      └── Manage State              ├── Input Collection
                                   ├── Validation
                                   └── Processing
```

### Data Processing Pipeline
```
1. Input Handling
   User Input ───> Parameter Object ───> Workflow Input
        │                │
        └── Validation──┘

2. Processing Stage
   ├── Model Configuration
   │      │
   │      ├── Required Parameters
   │      └── Optional Parameters
   │
   ├── Workflow Execution
   │      │
   │      ├── Progress Updates
   │      └── Error Handling
   │
   └── Results Generation
          │
          └── Data Models
              ├── Diffusivity Profile
              ├── Flux Data
              └── Fitting Results

3. Output Stage
   Results ───> Plot Generation ───> Display Update
      │              │                    │
      │              └── Four Panel Plot  ├── Progress Bar
      │                                   ├── Status Text
      └──────────────────────────────────>└── Results Text
```

### File Data Flow (Fitting Mode)
```
Data Selection
├── Preset Files Directory
│     │
│     └── Excel Files ───> File List ───> ComboBox
│
├── File Browser
│     │
│     └── Local Files ───> File Path ───> Entry Field
│
└── Selected File
      │
      ├── Auto Parameter Population
      │     ├── Thickness
      │     ├── Flowrate
      │     └── Default Diameter
      │
      └── Data Loading
            ├── Experimental Data
            └── Time Series
```

### User Interface Flow
```
1. Model Selection (Sidebar)
   ├── Load appropriate plugin
   └── Show default mode

2. Mode Selection
   ├── Switch between Manual/Fitting
   └── Preserve frame states

3. Data Processing
   ├── Input Collection
   ├── Validation
   ├── Processing
   └── Results Display
```

## Future Enhancements

1. Plugin Management
   - Dynamic plugin loading
   - Plugin configuration system
   - Plugin dependencies

2. Data Management
   - Session persistence
   - Result export options
   - Data preprocessing tools

3. UI Improvements
   - Advanced plotting options
   - Real-time parameter validation
   - Batch processing support
