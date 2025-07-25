#!/usr/bin/env python3
"""
Traffic Light Detection - Phase 3 Launcher
Simplified launcher script to run Phase 3 from project root
"""

import sys
import os
from pathlib import Path

# Add all necessary paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "phase3"))

# Change to project root directory
os.chdir(project_root)

# Import and run the phase3 script
if __name__ == "__main__":
    from src.phase3.phase3_execute import main
    main()