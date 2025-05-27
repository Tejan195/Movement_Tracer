# Launcher script
from src.utils.helpers import ensure_directories
from src.main import main

if __name__ == "__main__":
    # Ensure all directories exist
    ensure_directories()
    # Run the main application (now uses JumpRopeTracker as per main.py)
    main()
