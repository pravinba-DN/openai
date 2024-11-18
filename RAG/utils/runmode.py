import sys
import os

print(hasattr(sys, 'ps1'))

if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:  
    # Running in interactive mode (e.g., Jupyter or IPython)
    print("Interactive MODE")
elif 'VSCODE_PID' in os.environ:  
    # Running in VSCode terminal (based on the environment variable)
    print("Vscode")
else:
    # Default (use matplotlib for non-interactive environments like regular Python script)
    print("Default Terminal")