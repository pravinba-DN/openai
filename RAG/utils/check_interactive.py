import sys

if hasattr(sys, 'ps1'):
    print("Running in interactive mode")
else:
    print("Not running in interactive mode")
