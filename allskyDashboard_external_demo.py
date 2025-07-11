#!/usr/bin/env python3
"""
AllSkyCam Dashboard - External Data Sources Demo
Demonstrates how to feed GPS, IMU, and image data to the dashboard from external sources.
"""

import sys
import os
import threading
import time
import math
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class MockGPSHandler:
    """Mock GPS data source."""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.running = False
        
    def start(self):
        """Start GPS data updates."""
        self.running = True
        thread = threading.Thread(target=self._gps_loop, daemon=True)
        thread.start()
        return thread
        
    def stop(self):
        """Stop GPS data updates."""
        self.running = False
        
    def _gps_loop(self):
        """Simulate GPS updates."""
        start_time = time.time()
        
        while self.running:
            current_time = time.time() - start_time
            
            # Simulate GPS data
            latitude = 25.033 + 0.001 * math.sin(current_time * 0.1)
            longitude = 121.565 + 0.001 * math.cos(current_time * 0.1) 
            altitude = 150.0 + 20 * math.sin(current_time * 0.05)
            satellites = max(4, int(9 + 3 * math.sin(current_time * 0.2)))
            fix_quality = "GPS Fix" if (current_time % 15) < 12 else "No Fix"
            heading = (current_time * 5) % 360  # 5 deg/sec rotation
            
            # Send to dashboard
            self.dashboard.update_gps_data(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                satellites=satellites,
                fix_quality=fix_quality,
                heading=heading
            )
            
            time.sleep(0.5)  # Update at 2Hz

class MockIMUHandler:
    """Mock IMU data source."""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.running = False
        
    def start(self):
        """Start IMU data updates."""
        self.running = True
        thread = threading.Thread(target=self._imu_loop, daemon=True)
        thread.start()
        return thread
        
    def stop(self):
        """Stop IMU data updates."""
        self.running = False
        
    def _imu_loop(self):
        """Simulate IMU updates."""
        start_time = time.time()
        
        while self.running:
            current_time = time.time() - start_time
            
            # Simulate accelerometer data (some vibration + gravity)
            accel_x = 0.2 * math.sin(current_time * 8) + 0.1 * math.sin(current_time * 15)
            accel_y = 0.2 * math.cos(current_time * 8) + 0.1 * math.cos(current_time * 12)
            accel_z = 9.81 + 0.3 * math.sin(current_time * 3)  # Gravity + some noise
            
            # Simulate gyroscope data (slow rotation)
            gyro_x = 2.0 * math.sin(current_time * 0.8)
            gyro_y = 1.5 * math.cos(current_time * 0.6)
            gyro_z = 0.8 * math.sin(current_time * 0.3)
            
            # Simulate magnetometer data (Earth's magnetic field + interference)
            mag_x = 28.0 + 3 * math.sin(current_time * 0.4)
            mag_y = 35.0 + 4 * math.cos(current_time * 0.3)
            mag_z = 48.0 + 2 * math.sin(current_time * 0.2)
            
            # Simulate temperature
            temperature = 23.5 + 4 * math.sin(current_time * 0.05) + 0.5 * math.sin(current_time * 0.8)
            
            # Send to dashboard
            self.dashboard.update_imu_data(
                accel_x=accel_x, accel_y=accel_y, accel_z=accel_z,
                gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
                mag_x=mag_x, mag_y=mag_y, mag_z=mag_z,
                temperature=temperature
            )
            
            time.sleep(0.1)  # Update at 10Hz

class MockSunTracker:
    """Mock sun tracking data source."""
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.running = False
        
    def start(self):
        """Start sun tracking updates."""
        self.running = True
        thread = threading.Thread(target=self._sun_loop, daemon=True)
        thread.start()
        return thread
        
    def stop(self):
        """Stop sun tracking updates."""
        self.running = False
        
    def _sun_loop(self):
        """Simulate sun tracking updates."""
        start_time = time.time()
        
        while self.running:
            current_time = time.time() - start_time
            
            # Simulate sun position (slower movement, opposite direction)
            sun_heading = (360 - current_time * 2) % 360  # 2 deg/sec counter-rotation
            
            # Send to dashboard
            self.dashboard.update_sun_heading(sun_heading)
            
            time.sleep(1.0)  # Update at 1Hz

def main():
    """Demonstrate dashboard with multiple external data sources."""
    print("AllSkyCam Dashboard - External Data Sources Demo")
    print("=" * 52)
    
    try:
        from dashboard import Dashboard
        from fits_handler import FitsImageHandler
        
        # Create dashboard
        dashboard = Dashboard()
        
        # Create mock data sources
        gps_handler = MockGPSHandler(dashboard)
        imu_handler = MockIMUHandler(dashboard)
        sun_tracker = MockSunTracker(dashboard)
        
        # Create image source
        fits_handler = FitsImageHandler("images")
        
        print("Starting external data sources...")
        
        # Start all data sources
        gps_thread = gps_handler.start()
        imu_thread = imu_handler.start()
        sun_thread = sun_tracker.start()
        
        print("Data sources started:")
        print("- GPS Handler: 2Hz updates (lat/lon/alt/sats/fix/heading)")
        print("- IMU Handler: 10Hz updates (accel/gyro/mag/temp)")
        print("- Sun Tracker: 1Hz updates (sun heading)")
        
        if fits_handler.get_file_count() > 0:
            print(f"- Image Source: {fits_handler.get_file_count()} FITS files")
            
            # Start image feed
            def image_feed():
                while dashboard.running:
                    image_data, filename = fits_handler.next_image()
                    if image_data is not None:
                        info = f"Live: {filename} ({fits_handler.get_current_index() + 1}/{fits_handler.get_file_count()})"
                        dashboard.update_image_data(image_data, info)
                    time.sleep(3.0)  # 3-second intervals
                    
            image_thread = threading.Thread(target=image_feed, daemon=True)
            image_thread.start()
            print("- Image feed: 3-second intervals")
        
        print("\nDashboard features:")
        print("- Real GPS data from MockGPSHandler")
        print("- Real IMU data from MockIMUHandler") 
        print("- Real sun tracking from MockSunTracker")
        print("- Real image data from FitsImageHandler")
        print("- All data sources independent and modular")
        print("\nPress Ctrl+C or close window to exit.")
        
        # Run dashboard
        dashboard.run()
        
        # Cleanup
        gps_handler.stop()
        imu_handler.stop()
        sun_tracker.stop()
        
        print("External data sources demo completed.")
        return 0
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        return 0
    except Exception as e:
        print(f"Demo error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
