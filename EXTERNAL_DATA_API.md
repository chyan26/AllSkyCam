# Dashboard External Data API

The AllSkyCam Dashboard can accept data from external sources through dedicated update methods.

## Dashboard Class Methods

### Image Data
```python
dashboard.update_image_data(image_data, image_info="External image")
```
- **image_data**: numpy array or PIL Image
- **image_info**: string description of the image
- **Example**: 
  ```python
  dashboard.update_image_data(camera_frame, "Live Camera Feed")
  ```

### GPS Data
```python
dashboard.update_gps_data(latitude=None, longitude=None, altitude=None, 
                         satellites=None, fix_quality=None, heading=None)
```
- **latitude**: Latitude in degrees
- **longitude**: Longitude in degrees  
- **altitude**: Altitude in meters
- **satellites**: Number of satellites
- **fix_quality**: GPS fix quality string
- **heading**: GPS heading in degrees
- **Example**:
  ```python
  dashboard.update_gps_data(
      latitude=25.033,
      longitude=121.565,
      altitude=150.0,
      satellites=8,
      fix_quality="GPS Fix",
      heading=45.0
  )
  ```

### IMU Data
```python
dashboard.update_imu_data(accel_x=None, accel_y=None, accel_z=None,
                         gyro_x=None, gyro_y=None, gyro_z=None,
                         mag_x=None, mag_y=None, mag_z=None, temperature=None)
```
- **accel_x, accel_y, accel_z**: Accelerometer values in m/s²
- **gyro_x, gyro_y, gyro_z**: Gyroscope values in °/s
- **mag_x, mag_y, mag_z**: Magnetometer values in µT
- **temperature**: Temperature in °C
- **Example**:
  ```python
  dashboard.update_imu_data(
      accel_x=0.1, accel_y=0.2, accel_z=9.8,
      gyro_x=1.5, gyro_y=-0.8, gyro_z=0.3,
      mag_x=28.0, mag_y=35.0, mag_z=48.0,
      temperature=25.5
  )
  ```

### Sun Tracking Data
```python
dashboard.update_sun_heading(heading)
```
- **heading**: Sun heading in degrees
- **Example**:
  ```python
  dashboard.update_sun_heading(180.5)
  ```

## Integration Examples

### With Real GPS Handler
```python
from dashboard import Dashboard
from gpsHandler import GPSHandler

dashboard = Dashboard()
gps = GPSHandler()

# Update dashboard with real GPS data
def gps_callback(gps_data):
    dashboard.update_gps_data(
        latitude=gps_data.latitude,
        longitude=gps_data.longitude,
        altitude=gps_data.altitude,
        satellites=gps_data.satellites,
        fix_quality=gps_data.fix_quality,
        heading=gps_data.heading
    )

gps.set_callback(gps_callback)
```

### With Real Camera
```python
from dashboard import Dashboard
from allskyController import AllSkyController

dashboard = Dashboard()
camera = AllSkyController()

# Update dashboard with camera frames
def camera_callback(image_data):
    dashboard.update_image_data(image_data, "Live AllSkyCam Feed")

camera.set_frame_callback(camera_callback)
```

### With Sun Detection
```python
from dashboard import Dashboard
from detectSun import ImageProcessor

dashboard = Dashboard()
sun_detector = ImageProcessor()

# Update dashboard with sun position
def sun_callback(sun_azimuth):
    dashboard.update_sun_heading(sun_azimuth)

sun_detector.set_detection_callback(sun_callback)
```

## Complete Integration Example
```python
from dashboard import Dashboard
from gpsHandler import GPSHandler
from allskyController import AllSkyController
from detectSun import ImageProcessor

# Create dashboard
dashboard = Dashboard()

# Create data sources
gps = GPSHandler()
camera = AllSkyController()
sun_detector = ImageProcessor()

# Set up data flow
def update_gps(data):
    dashboard.update_gps_data(
        latitude=data.latitude,
        longitude=data.longitude,
        heading=data.heading
        # ... other GPS fields
    )

def update_camera(image):
    dashboard.update_image_data(image, "AllSkyCam Live")
    
    # Also process for sun detection
    sun_azimuth = sun_detector.detect_sun(image)
    if sun_azimuth:
        dashboard.update_sun_heading(sun_azimuth)

def update_imu(imu_data):
    dashboard.update_imu_data(
        accel_x=imu_data.accel[0],
        accel_y=imu_data.accel[1], 
        accel_z=imu_data.accel[2],
        # ... other IMU fields
    )

# Connect callbacks
gps.set_callback(update_gps)
camera.set_frame_callback(update_camera)
# imu.set_callback(update_imu)  # When IMU is available

# Run dashboard
dashboard.run()
```

## Benefits of External Data Design

✅ **Modular**: Dashboard is decoupled from data sources  
✅ **Flexible**: Easy to swap data sources or add new ones  
✅ **Testable**: Can use mock data sources for testing  
✅ **Real-time**: Each data source can update at its own rate  
✅ **Thread-safe**: Dashboard handles concurrent updates safely  

This design allows the dashboard to be a pure display component that can work with any data source!
