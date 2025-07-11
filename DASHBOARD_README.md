# AllSkyCam Dashboard

A clean, modular dashboard for displaying AllSkyCam system data in real-time.

## Overview

The dashboard is a **pure display component** that accepts data from external sources. It provides real-time visualization of GPS, IMU, image, and sun tracking data in a compact 640x480 window.

## Features

### Clean Architecture
- **No built-in data sources**: Dashboard only displays data
- **External data API**: Simple methods to update all data types
- **Modular design**: Easy integration with any data source
- **Thread-safe**: Safe for multi-threaded applications

### Display Components (640x480 layout)
- **Image Preview**: 200x200 square with aspect ratio preservation
- **Compass Display**: GPS and sun heading vectors with degree markings
- **GPS Data**: Coordinates, altitude, satellites, fix quality
- **IMU Data**: Tabbed interface for accelerometer/gyroscope/magnetometer
- **Temperature**: Real-time sensor temperature display

## Quick Start

### Basic Usage
```bash
python dashboard.py
```

### With Sample Data
```bash
python allskyDashboard_basic_demo.py
```

### With External Data Sources
```bash
python allskyDashboard_external_demo.py
```

## External Data API

### Update Image Data
```python
dashboard.update_image_data(image_array, "Camera Feed")
```

### Update GPS Data  
```python
dashboard.update_gps_data(
    latitude=25.033, longitude=121.565, altitude=150.0,
    satellites=8, fix_quality="GPS Fix", heading=45.0
)
```

### Update IMU Data
```python
dashboard.update_imu_data(
    accel_x=0.1, accel_y=0.2, accel_z=9.8,
    gyro_x=1.5, gyro_y=-0.8, gyro_z=0.3,
    mag_x=28.0, mag_y=35.0, mag_z=48.0,
    temperature=25.5
)
```

### Update Sun Heading
```python
dashboard.update_sun_heading(180.5)
```

## Integration with AllSkyCam

### With GPS Handler
```python
from dashboard import Dashboard
from gpsHandler import GPSHandler

dashboard = Dashboard()
gps = GPSHandler()

def gps_callback(data):
    dashboard.update_gps_data(
        latitude=data.latitude,
        longitude=data.longitude,
        heading=data.heading
    )

gps.set_callback(gps_callback)
```

### With Camera Controller
```python
from allskyController import AllSkyController

camera = AllSkyController()

def camera_callback(image):
    dashboard.update_image_data(image, "Live Feed")

camera.set_frame_callback(camera_callback)
```

## File Structure

### Core Files
- **`dashboard.py`** - Main dashboard class (pure display)
- **`fits_handler.py`** - Separate FITS file management
- **`allskyDashboard_basic_demo.py`** - Basic demonstration with sample data
- **`allskyDashboard_external_demo.py`** - Advanced demo with mock data sources

### Documentation
- **`DASHBOARD_README.md`** - This file
- **`EXTERNAL_DATA_API.md`** - Detailed API documentation

## Dependencies

- Python 3.8+
- tkinter (usually included)
- numpy
- Pillow (PIL)
- astropy (for FITS support in fits_handler)

```bash
pip install numpy Pillow astropy
```

## Design Benefits

✅ **Clean separation**: Dashboard vs data management  
✅ **Easy testing**: Use mock data sources  
✅ **Flexible integration**: Works with any data source  
✅ **Real-time ready**: 10Hz GUI updates  
✅ **Thread-safe**: Concurrent data updates supported  

The dashboard is now ready for seamless integration with your AllSkyCam hardware!
