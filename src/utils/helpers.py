import cv2
import numpy as np
from datetime import datetime
import os

def ensure_directories():
    """Ensure all required directories exist"""
    dirs = ['data/logs', 'src/utils', 'src/models', 'src/views']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_session_stats(stats, user_info):
    """Save session statistics to a log file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/logs/session_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Session: {timestamp}\n")
        f.write(f"User: Weight={user_info['weight']}kg, Height={user_info['height']}cm, Gender={user_info['gender']}\n")
        f.write(f"Stats: {stats}\n")
    
    return filename
