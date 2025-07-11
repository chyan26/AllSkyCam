#!/usr/bin/env python3
"""
AllSkyCam Dashboard

A clean, modular dashboard for displaying AllSkyCam data.
Accepts data from external sources via update methods.

Features:
- Image preview with aspect ratio preservation
- GPS data display and heading vector
- IMU data display (accelerometer, gyroscope, magnetometer) - all data visible simultaneously 
- Sun tracking heading vector
- Real-time updates at 10Hz
- 640x480 pixel window size

External Data API:
- update_image_data(image_data, info)
- update_gps_data(lat, lon, alt, sats, fix, heading)
- update_imu_data(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, temp)
- update_sun_heading(heading)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import math
import numpy as np
from PIL import Image, ImageTk
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dashboard:
    """Main dashboard class for AllSkyCam monitoring."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AllSkyCam Dashboard")
        self.root.geometry("640x480")
        self.root.resizable(False, False)
        
        # Data storage
        self.current_image = None
        self.gps_heading = 0.0
        self.sun_heading = 0.0
        self.gps_data = {
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0,
            'satellites': 0,
            'fix_quality': 'No Fix'
        }
        self.imu_data = {
            'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
            'mag_x': 0.0, 'mag_y': 0.0, 'mag_z': 0.0,
            'temperature': 25.0
        }
        
        # Threading control
        self.running = True
        
        self.setup_ui()
        self.start_gui_updates()
        
    def setup_ui(self):
        """Setup the user interface layout."""
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Top section - Image preview (left) and Heading vectors (right)
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True)
        
        # Image preview section
        self.setup_image_section(top_frame)
        
        # Heading vectors section  
        self.setup_heading_section(top_frame)
        
        # Bottom section - GPS and IMU data
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.setup_gps_section(bottom_frame)
        self.setup_imu_section(bottom_frame)
        
        # Control buttons
        self.setup_controls(main_frame)
        
    def setup_image_section(self, parent):
        """Setup image preview section."""
        image_frame = ttk.LabelFrame(parent, text="Image Preview", padding=5)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        
        # Image display canvas - smaller size to save space
        self.image_canvas = tk.Canvas(image_frame, width=200, height=200, bg='black')
        self.image_canvas.pack()
        
        # Image info
        self.image_info = ttk.Label(image_frame, text="No image loaded")
        self.image_info.pack(pady=2)
        
        # Load image button
        load_btn = ttk.Button(image_frame, text="Load Image", command=self.load_image)
        load_btn.pack(pady=2)
        
    def setup_heading_section(self, parent):
        """Setup heading vectors visualization."""
        heading_frame = ttk.LabelFrame(parent, text="Heading Vectors", padding=5)
        heading_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))
        
        # Compass canvas - smaller size to save space
        self.compass_canvas = tk.Canvas(heading_frame, width=200, height=200, bg='white')
        self.compass_canvas.pack()
        
        # Draw compass base
        self.draw_compass_base()
        
        # Heading values
        values_frame = ttk.Frame(heading_frame)
        values_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(values_frame, text="GPS Heading:").grid(row=0, column=0, sticky=tk.W)
        self.gps_heading_label = ttk.Label(values_frame, text="0.0°", foreground="blue")
        self.gps_heading_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(values_frame, text="Sun Heading:").grid(row=1, column=0, sticky=tk.W)
        self.sun_heading_label = ttk.Label(values_frame, text="0.0°", foreground="red")
        self.sun_heading_label.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
    def setup_gps_section(self, parent):
        """Setup GPS data display."""
        gps_frame = ttk.LabelFrame(parent, text="GPS Data", padding=5)
        gps_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        
        # GPS data grid
        self.gps_labels = {}
        gps_fields = [
            ('Latitude:', 'latitude', '°'),
            ('Longitude:', 'longitude', '°'),
            ('Altitude:', 'altitude', 'm'),
            ('Satellites:', 'satellites', ''),
            ('Fix Quality:', 'fix_quality', '')
        ]
        
        for i, (label, key, unit) in enumerate(gps_fields):
            ttk.Label(gps_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=1)
            self.gps_labels[key] = ttk.Label(gps_frame, text=f"0.0{unit}")
            self.gps_labels[key].grid(row=i, column=1, sticky=tk.W, padx=(5, 0), pady=1)
            
    def setup_imu_section(self, parent):
        """Setup IMU data display - all data visible without tabs."""
        imu_frame = ttk.LabelFrame(parent, text="IMU Data", padding=5)
        imu_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))
        
        # Initialize labels dictionary
        if not hasattr(self, 'imu_labels'):
            self.imu_labels = {}
        
        # Create a compact grid layout for all IMU data
        # Row 0: Headers
        ttk.Label(imu_frame, text="Accel (m/s²)", font=('TkDefaultFont', 8, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0,2))
        ttk.Label(imu_frame, text="Gyro (°/s)", font=('TkDefaultFont', 8, 'bold')).grid(row=0, column=2, columnspan=2, sticky=tk.W, padx=(10,0), pady=(0,2))
        ttk.Label(imu_frame, text="Mag (µT)", font=('TkDefaultFont', 8, 'bold')).grid(row=0, column=4, columnspan=2, sticky=tk.W, padx=(10,0), pady=(0,2))
        
        # Accelerometer data (rows 1-3, columns 0-1)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ttk.Label(imu_frame, text=f"{axis}:", font=('TkDefaultFont', 8)).grid(row=i+1, column=0, sticky=tk.W)
            field = f'accel_{axis.lower()}'
            self.imu_labels[field] = ttk.Label(imu_frame, text="0.0", font=('TkDefaultFont', 8))
            self.imu_labels[field].grid(row=i+1, column=1, sticky=tk.W, padx=(2,0))
        
        # Gyroscope data (rows 1-3, columns 2-3)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ttk.Label(imu_frame, text=f"{axis}:", font=('TkDefaultFont', 8)).grid(row=i+1, column=2, sticky=tk.W, padx=(10,0))
            field = f'gyro_{axis.lower()}'
            self.imu_labels[field] = ttk.Label(imu_frame, text="0.0", font=('TkDefaultFont', 8))
            self.imu_labels[field].grid(row=i+1, column=3, sticky=tk.W, padx=(2,0))
        
        # Magnetometer data (rows 1-3, columns 4-5)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ttk.Label(imu_frame, text=f"{axis}:", font=('TkDefaultFont', 8)).grid(row=i+1, column=4, sticky=tk.W, padx=(10,0))
            field = f'mag_{axis.lower()}'
            self.imu_labels[field] = ttk.Label(imu_frame, text="0.0", font=('TkDefaultFont', 8))
            self.imu_labels[field].grid(row=i+1, column=5, sticky=tk.W, padx=(2,0))
        
        # Temperature (row 4, spans across columns)
        ttk.Label(imu_frame, text="Temperature:", font=('TkDefaultFont', 8, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(5,0))
        self.temp_label = ttk.Label(imu_frame, text="25.0°C", font=('TkDefaultFont', 8))
        self.temp_label.grid(row=4, column=1, columnspan=2, sticky=tk.W, padx=(2,0), pady=(5,0))
            
    def setup_controls(self, parent):
        """Setup control buttons."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Reset data
        reset_btn = ttk.Button(control_frame, text="Reset Data", command=self.reset_data)
        reset_btn.pack(side=tk.LEFT)
        
        # Exit button
        exit_btn = ttk.Button(control_frame, text="Exit", command=self.on_closing)
        exit_btn.pack(side=tk.RIGHT)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready for external data")
        self.status_label.pack(side=tk.RIGHT, padx=(0, 10))
        
    def draw_compass_base(self):
        """Draw the compass base with cardinal directions."""
        canvas = self.compass_canvas
        canvas.delete("compass_base")
        
        center_x, center_y = 100, 100  # Centered in 200x200 canvas
        radius = 70
        
        # Draw compass circle
        canvas.create_oval(center_x - radius, center_y - radius,
                          center_x + radius, center_y + radius,
                          outline="black", width=2, tags="compass_base")
        
        # Draw cardinal directions with degrees
        directions = [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")]
        for angle, label in directions:
            x = center_x + (radius + 15) * math.sin(math.radians(angle))
            y = center_y - (radius + 15) * math.cos(math.radians(angle))
            canvas.create_text(x, y, text=label, font=("Arial", 10, "bold"),
                             tags="compass_base")
            
        # Draw center point
        canvas.create_oval(center_x - 3, center_y - 3, center_x + 3, center_y + 3,
                          fill="black", tags="compass_base")
                          
    def update_compass(self):
        """Update compass with current heading vectors."""
        canvas = self.compass_canvas
        canvas.delete("heading_vectors")
        
        center_x, center_y = 100, 100  # Centered in 200x200 canvas
        radius = 60
        
        # Draw GPS heading (blue arrow)
        if self.gps_heading is not None:
            angle_rad = math.radians(self.gps_heading)
            end_x = center_x + radius * math.sin(angle_rad)
            end_y = center_y - radius * math.cos(angle_rad)
            canvas.create_line(center_x, center_y, end_x, end_y,
                             fill="blue", width=3, arrow=tk.LAST,
                             arrowshape=(10, 12, 3), tags="heading_vectors")
            
        # Draw Sun heading (red arrow)
        if self.sun_heading is not None:
            angle_rad = math.radians(self.sun_heading)
            end_x = center_x + (radius - 10) * math.sin(angle_rad)
            end_y = center_y - (radius - 10) * math.cos(angle_rad)
            canvas.create_line(center_x, center_y, end_x, end_y,
                             fill="red", width=3, arrow=tk.LAST,
                             arrowshape=(8, 10, 3), tags="heading_vectors")
                             
    def update_image_data(self, image_data, image_info="External image"):
        """Update the image display with new image data from external source.
        
        Args:
            image_data: numpy array or PIL Image
            image_info: string description of the image
        """
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image_data, np.ndarray):
                # Handle different data types and normalize for display
                if image_data.dtype != np.uint8:
                    # Normalize to 0-255 range
                    image_min, image_max = image_data.min(), image_data.max()
                    if image_max > image_min:
                        image_data = ((image_data - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                    else:
                        image_data = np.zeros_like(image_data, dtype=np.uint8)
                
                image = Image.fromarray(image_data)
            else:
                image = image_data
                
            # Resize maintaining aspect ratio
            original_width, original_height = image.size
            canvas_size = 200
            
            if original_width == original_height:
                # Square image
                image = image.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
            else:
                # Non-square image - maintain aspect ratio
                if original_width > original_height:
                    new_width = canvas_size
                    new_height = int((original_height * canvas_size) / original_width)
                else:
                    new_height = canvas_size
                    new_width = int((original_width * canvas_size) / original_height)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            self.current_image = ImageTk.PhotoImage(image)
            self.image_canvas.delete("all")
            self.image_canvas.create_image(100, 100, image=self.current_image)
            
            self.image_info.config(text=image_info)
            logger.info(f"Updated image: {image_info}")
            
        except Exception as e:
            logger.error(f"Failed to update image data: {e}")
            self.image_info.config(text=f"Error: {e}")
            
    def update_gps_data(self, latitude=None, longitude=None, altitude=None, 
                       satellites=None, fix_quality=None, heading=None):
        """Update GPS data from external source.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees  
            altitude: Altitude in meters
            satellites: Number of satellites
            fix_quality: GPS fix quality string
            heading: GPS heading in degrees
        """
        if latitude is not None:
            self.gps_data['latitude'] = latitude
        if longitude is not None:
            self.gps_data['longitude'] = longitude
        if altitude is not None:
            self.gps_data['altitude'] = altitude
        if satellites is not None:
            self.gps_data['satellites'] = satellites
        if fix_quality is not None:
            self.gps_data['fix_quality'] = fix_quality
        if heading is not None:
            self.gps_heading = heading
            
        logger.debug("Updated GPS data from external source")
        
    def update_imu_data(self, accel_x=None, accel_y=None, accel_z=None,
                       gyro_x=None, gyro_y=None, gyro_z=None,
                       mag_x=None, mag_y=None, mag_z=None, temperature=None):
        """Update IMU data from external source.
        
        Args:
            accel_x, accel_y, accel_z: Accelerometer values in m/s²
            gyro_x, gyro_y, gyro_z: Gyroscope values in °/s
            mag_x, mag_y, mag_z: Magnetometer values in µT
            temperature: Temperature in °C
        """
        if accel_x is not None:
            self.imu_data['accel_x'] = accel_x
        if accel_y is not None:
            self.imu_data['accel_y'] = accel_y
        if accel_z is not None:
            self.imu_data['accel_z'] = accel_z
        if gyro_x is not None:
            self.imu_data['gyro_x'] = gyro_x
        if gyro_y is not None:
            self.imu_data['gyro_y'] = gyro_y
        if gyro_z is not None:
            self.imu_data['gyro_z'] = gyro_z
        if mag_x is not None:
            self.imu_data['mag_x'] = mag_x
        if mag_y is not None:
            self.imu_data['mag_y'] = mag_y
        if mag_z is not None:
            self.imu_data['mag_z'] = mag_z
        if temperature is not None:
            self.imu_data['temperature'] = temperature
            
        logger.debug("Updated IMU data from external source")
        
    def update_sun_heading(self, heading):
        """Update sun tracking heading from external source.
        
        Args:
            heading: Sun heading in degrees
        """
        self.sun_heading = heading
        logger.debug(f"Updated sun heading: {heading}°")
        
    def load_image(self):
        """Load an image file for preview."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.fits"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Handle FITS files
                if file_path.lower().endswith('.fits'):
                    try:
                        from astropy.io import fits
                        with fits.open(file_path) as hdul:
                            image_data = hdul[0].data
                            # Normalize for display
                            image_data = ((image_data - image_data.min()) / 
                                        (image_data.max() - image_data.min()) * 255).astype(np.uint8)
                            image = Image.fromarray(image_data)
                    except ImportError:
                        messagebox.showerror("Error", "astropy required for FITS file support")
                        return
                else:
                    image = Image.open(file_path)
                    
                # Maintain aspect ratio when resizing
                original_width, original_height = image.size
                canvas_size = 200  # Smaller canvas size
                
                if original_width == original_height:
                    # Square image - fits perfectly
                    image = image.resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)
                else:
                    # Non-square image - fit within square maintaining aspect ratio
                    if original_width > original_height:
                        new_width = canvas_size
                        new_height = int((original_height * canvas_size) / original_width)
                    else:
                        new_height = canvas_size
                        new_width = int((original_width * canvas_size) / original_height)
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                self.current_image = ImageTk.PhotoImage(image)
                self.image_canvas.delete("all")
                self.image_canvas.create_image(100, 100, image=self.current_image)
                
                filename = os.path.basename(file_path)
                self.image_info.config(text=f"Loaded: {filename}")
                logger.info(f"Loaded image: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                logger.error(f"Failed to load image {file_path}: {e}")
                
    def start_gui_updates(self):
        """Start the GUI update loop."""
        self.update_gui()
        
    def update_gui(self):
        """Update GUI elements with current data."""
        if not self.running:
            return
            
        # Update heading labels
        self.gps_heading_label.config(text=f"{self.gps_heading:.1f}°")
        self.sun_heading_label.config(text=f"{self.sun_heading:.1f}°")
        
        # Update compass
        self.update_compass()
        
        # Update GPS data
        for key, label in self.gps_labels.items():
            value = self.gps_data[key]
            if key in ['latitude', 'longitude', 'altitude']:
                label.config(text=f"{value:.6f}")
            elif key == 'satellites':
                label.config(text=str(int(value)))
            else:
                label.config(text=str(value))
                
        # Update IMU data
        for key, label in self.imu_labels.items():
            value = self.imu_data[key]
            label.config(text=f"{value:.2f}")
            
        # Update temperature
        self.temp_label.config(text=f"{self.imu_data['temperature']:.1f}°C")
        
        # Schedule next update
        self.root.after(100, self.update_gui)  # Update GUI at 10Hz
        
    def reset_data(self):
        """Reset all data to default values."""
        self.gps_heading = 0.0
        self.sun_heading = 0.0
        self.gps_data = {
            'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0,
            'satellites': 0, 'fix_quality': 'No Fix'
        }
        self.imu_data = {
            'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
            'mag_x': 0.0, 'mag_y': 0.0, 'mag_z': 0.0,
            'temperature': 25.0
        }
        logger.info("Data reset to defaults")
        
    def on_closing(self):
        """Handle application closing."""
        self.running = False
        self.root.destroy()
        
    def run(self):
        """Start the dashboard application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        logger.info("Starting AllSkyCam Dashboard")
        self.root.mainloop()

def main():
    """Main function to run the dashboard standalone."""
    try:
        dashboard = Dashboard()
        dashboard.run()
    except KeyboardInterrupt:
        logger.info("Dashboard interrupted by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise

if __name__ == "__main__":
    main()
