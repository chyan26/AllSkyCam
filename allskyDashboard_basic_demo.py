#!/usr/bin/env python3
"""
AllSkyCam Dashboard - Basic Demo
Simple demonstration of the dashboard with minimal external data examples.
"""

import sys
import os
import threading
import time
import math

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simple_demo():
    """Simple demonstration of dashboard capabilities."""
    try:
        from dashboard import Dashboard
        from fits_handler import FitsImageHandler
        
        print("AllSkyCam Dashboard - Simple Demo")
        print("=" * 35)
        
        # Create dashboard
        dashboard = Dashboard()
        
        # Optional: Load a FITS image if available
        fits_handler = FitsImageHandler("images")
        if fits_handler.get_file_count() > 0:
            image_data, filename = fits_handler.load_current_image()
            if image_data is not None:
                dashboard.update_image_data(image_data, f"Sample: {filename}")
                print(f"Loaded sample image: {filename}")
        
        # Set some sample data
        dashboard.update_gps_data(
            latitude=25.033,
            longitude=121.565,
            altitude=150.0,
            satellites=8,
            fix_quality="GPS Fix",
            heading=45.0
        )
        
        dashboard.update_imu_data(
            accel_x=0.1, accel_y=0.2, accel_z=9.8,
            gyro_x=1.5, gyro_y=-0.8, gyro_z=0.3,
            mag_x=28.0, mag_y=35.0, mag_z=48.0,
            temperature=25.5
        )
        
        dashboard.update_sun_heading(180.5)
        
        print("Dashboard ready with sample data")
        print("Use 'Load Image' button to load additional images")
        print("Press Ctrl+C or close window to exit.")
        
        # Run dashboard
        dashboard.run()
        
        print("Demo completed.")
        
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    simple_demo()
