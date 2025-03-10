import os
import glob
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from datetime import datetime
from detectSun import ImageProcessor  # Import the ImageProcessor class

def extract_lat_lon_azi_from_fits(file_path):
    with fits.open(file_path) as hdul:
        header = hdul[0].header
        latitude = header['LATITUDE']
        longitude = header['LONGITUD']
        azimuth = header.get('MEAS_AZI', 0)
        
    return latitude, longitude, azimuth

def parse_datetime_from_filename(filename):
    basename = os.path.basename(filename)
    try:
        date_str = basename.split('_')[1] + basename.split('_')[2] + basename.split('_')[3] + basename.split('_')[4]
        return datetime.strptime(date_str, '%Y%m%d%H%M%S')
    except (IndexError, ValueError) as e:
        print(f"Error parsing date from filename {filename}: {e}")
        return None

def collect_lat_lon_data(file_pattern, start_time, end_time):
    latitudes = []
    longitudes = []
    azimuths = []
    labels = []
    fits_files = glob.glob(file_pattern)
    fits_files.sort()
    print(f"Found {len(fits_files)} files matching the pattern.")
    
    for fits_file in fits_files:
        file_datetime = parse_datetime_from_filename(fits_file)
        if file_datetime is None:
            continue
        if start_time <= file_datetime <= end_time:
            lat, lon, azi = extract_lat_lon_azi_from_fits(fits_file)
            print(f"File {fits_file} has LATITUDE={lat}, LONGITUDE={lon}, AZIMUTH={azi}")
            if lat is not None and lon is not None:
                latitudes.append(lat)
                longitudes.append(lon)
                azimuths.append(azi)
                label = os.path.basename(fits_file).split('_')[-1].split('.')[0]
                labels.append(label)
            else:
                print(f"File {fits_file} does not contain LATITUDE or LONGITUDE in the header.")
    
    return latitudes, longitudes, azimuths, labels

def calculate_rotation_angle(measured_azimuth, actual_azimuth):
    rotation_angle = actual_azimuth - measured_azimuth
    return rotation_angle

def plot_lat_lon(latitudes, longitudes, labels, solar_azimuths):
    if not latitudes or not longitudes:
        print("No data to plot.")
        return
    
    plt.scatter(longitudes, latitudes, marker='o', color='b')
    
    # Calculate the range of the data
    lon_range = max(longitudes) - min(longitudes)
    lat_range = max(latitudes) - min(latitudes)
    
    # Set the arrow length as a fraction of the data range
    arrow_length = min(lon_range, lat_range) * 0.005  # Adjusted to 2% of the smaller range
    
    for i, label in enumerate(labels):
        plt.text(longitudes[i], latitudes[i], label, fontsize=9, ha='right')
    
    # Calculate vector components based on the direction to the next position
    u = []
    v = []
    for i in range(len(latitudes) - 1):
        delta_lon = longitudes[i + 1] - longitudes[i]
        delta_lat = latitudes[i + 1] - latitudes[i]
        norm = np.sqrt(delta_lon**2 + delta_lat**2)
        if norm != 0:
            u.append(arrow_length * (delta_lon / norm))
            v.append(arrow_length * (delta_lat / norm))
        else:
            u.append(0)
            v.append(0)
    
    # Add a zero vector for the last point
    u.append(0)
    v.append(0)
    
    #plt.quiver(longitudes, latitudes, u, v, angles='xy', scale_units='xy', scale=0.15, color='r')
    
    solar_azimuths = 90-np.array(solar_azimuths) % 360  # Convert to meteorological convention
    # Plot solar azimuths as vectors
    solar_u = []
    solar_v = []
    for solar_azi in solar_azimuths:
        solar_u.append(arrow_length * np.cos(np.radians(solar_azi)))
        solar_v.append(arrow_length * np.sin(np.radians(solar_azi)))
    
    plt.quiver(longitudes, latitudes, solar_u, solar_v, angles='xy', scale_units='xy', scale=0.15, color='g')
    
     
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Latitude vs Longitude with Direction Vectors and Solar Azimuths')
    plt.grid(True)
    plt.show()

def runAnalysis():
    file_pattern = os.path.expanduser("~/AllSkyCam/output/image_*.fits")
    start_time = datetime(2025, 2, 11, 14, 43, 31)  # Start time: 2025-02-11 14:43:31
    end_time = datetime(2025, 2, 11, 14, 49, 14)    # End time: 2025-02-11 14:49:14
    latitudes, longitudes, azimuths, labels = collect_lat_lon_data(file_pattern, start_time, end_time)
    print(f"Latitudes: {latitudes}")
    print(f"Longitudes: {longitudes}")
    print(f"Azimuths: {azimuths}")
    print(f"Labels: {labels}")

    fits_files = glob.glob(file_pattern)
    fits_files.sort()

    solar_azimuths = []
    image_arrays = []

    edges = None
    for fits_file in fits_files:
        file_datetime = parse_datetime_from_filename(fits_file)
        if file_datetime is None:
            continue
        if start_time <= file_datetime <= end_time:
            processor = ImageProcessor(fits_file)
            print(f"Processing image {fits_file}...")
            
            if edges is None:
                edges = processor.edgeDetection(display=False)
                allsky_x, allsky_y, allsky_r = edges[0,0], edges[0,1], edges[0,2]
                print(f"Edges detected in image {fits_file}: {edges}")

            processor.sunDetectionSEP()
            print(f"processor.sunLocation: {processor.sunLocation}")
            sun_x, sun_y, _ = processor.sunLocation
            print(f"Sun location detected in image {fits_file}: x={sun_x}, y={sun_y}")
            processor.calSunAltAzi((allsky_x, allsky_y), (sun_x, sun_y), allsky_r)
            sun_alt, sun_azi = processor.sunMeasuredAlt, processor.sunMeasuredAzi
            print(f"Sun altitude and azimuth in image {fits_file}: {sun_alt}, {sun_azi}")
            solar_azimuths.append(sun_azi)

            # Collect image data
            with fits.open(fits_file) as hdul:
                image_data = hdul[0].data
                image_arrays.append(image_data)

    #plot_lat_lon(latitudes, longitudes, labels, solar_azimuths)

    if solar_azimuths:
        measured_azimuth = np.array(solar_azimuths)
        actual_azimuth = 227.786+23.1593928
        rotation_angle = calculate_rotation_angle(measured_azimuth, actual_azimuth)
        print(f"Measured Azimuth: {measured_azimuth}")
        print(f"Actual Azimuth: {actual_azimuth}")
        print(f"Rotation Angle: {rotation_angle}")    

    plot_lat_lon(latitudes, longitudes, labels, rotation_angle)
if __name__ == "__main__":
    runAnalysis()
    