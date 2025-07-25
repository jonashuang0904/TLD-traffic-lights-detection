#!/usr/bin/env python3
"""
Traffic Light Detection - Main Evaluation Launcher
Simplified launcher script to run evaluation from project root
"""

import sys
import os
from pathlib import Path

# Add all necessary paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "phase4"))

# Change to project root directory
os.chdir(project_root)

# Import and run the evaluation script
if __name__ == "__main__":
    from src.phase4.model_evaluation import main
    main()