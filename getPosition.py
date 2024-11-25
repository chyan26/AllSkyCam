from skyfield.api import Topos, load
from datetime import datetime, timedelta
from scipy.optimize import minimize
import numpy as np

# Initialize the Skyfield data
planets = load('de421.bsp')
earth = planets['earth']
sun = planets['sun']
ts = load.timescale()

# Input data
altitude_deg = 65.0      # Altitude of the Sun in degrees
azimuth_deg = 180.0      # Azimuth (position angle) of the Sun in degrees, 0 = North, 90 = East
local_time = datetime(2024, 3, 20, 4, 0, 0)  # UTC time of observation (example date and time)

def compute_sun_alt_az(latitude, longitude):
    # Generate observer's location
    location = Topos(latitude_degrees=latitude, longitude_degrees=longitude)
    time = ts.utc(local_time.year, local_time.month, local_time.day,
                  local_time.hour, local_time.minute, local_time.second)
    
    # Compute Sun's apparent position from this location and time
    observer = earth + location
    astrometric = observer.at(time).observe(sun)
    alt, az, _ = astrometric.apparent().altaz()
    
    return alt.degrees, az.degrees

def objective(coords):
    latitude, longitude = coords
    computed_alt, computed_az = compute_sun_alt_az(latitude, longitude)
    alt_diff = (computed_alt - altitude_deg)**2
    az_diff = (computed_az - azimuth_deg)**2
    return alt_diff + az_diff

# Initial guess (example for somewhere near the equator)
initial_guess = [0.0, 0.0]  # latitude, longitude in degrees

# Minimize the objective function
result = minimize(objective, initial_guess, bounds=[(-90, 90), (-180, 180)])

# Extract latitude and longitude if successful
if result.success:
    latitude, longitude = result.x
    print(f"Latitude: {latitude:.6f}°, Longitude: {longitude:.6f}°")
else:
    print("Optimization failed.")
